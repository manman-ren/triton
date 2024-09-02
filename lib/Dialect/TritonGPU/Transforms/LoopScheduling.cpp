#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace mlir {
namespace triton {
namespace gpu {

static void scheduleLoads(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                          DenseSet<Operation *> &rootUsers, int numStages) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse =
          mlir::triton::loadOpsToIndirectionLevelAndUse(forOp);
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return;

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      rootUsers.insert(use);
      schedule.insert(use, numStages - 1, rootUsersCluster);
    }
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }
}

static tt::CoarseSchedule::Cluster
schedulePrologueAndEpilogue(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                            DenseSet<Operation *> &rootUsers, int numStages) {
  // afterPrologue : first cluster curently but we will add a cluster at front
  // and a cluster at back
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  tt::CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }
  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

static const char *kLoopScheduleAttrName = "tt.loop_schedule";
std::string getLoopScheduleOrDefault(scf::ForOp forOp) {
  if (!forOp->hasAttr(kLoopScheduleAttrName))
    return "default";
  return (cast<StringAttr>(forOp->getAttr(kLoopScheduleAttrName))).str();
}

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }

  tt::CoarseSchedule::Cluster
  getDefaultLoopSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                         int numStages) {
    DenseSet<Operation *> rootUsers;
    scheduleLoads(forOp, schedule, rootUsers, numStages);
    return schedulePrologueAndEpilogue(forOp, schedule, rootUsers, numStages);
  }

  bool isFlashAttention(scf::ForOp forOp, SmallVector<Operation *> &keyOps) {
    SmallVector<Operation *> loads;
    SmallVector<Operation *> dots;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // Check for loop-carried dependencies.
      // We have two loadOps, one feeding the first dot, and the other feeding
      // the second dot.
      if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
        loads.push_back(&op);
      }
      if (op.hasTrait<OpTrait::DotLike>()) {
        dots.push_back(&op);
      }
    }
    if (dots.size() != 2 || loads.size() != 2)
      return false;

    Operation *secondDot = dots[1];
    DenseSet<Operation *> seen;
    DenseSet<Operation *> tracedDots;
    // Make sure there is a dependency path from firstDot to secondDot.
    // This means we need to do computation pipelining to break the dependency.
    std::function<void(Operation * op)> dfs = [&](Operation *op) {
      if (!seen.insert(op).second)
        return;
      for (Value operand : op->getOperands()) {
        Value v = operand;
        Operation *defOp = v.getDefiningOp();
        if (defOp && defOp->getBlock() == op->getBlock()) {
          if (defOp->hasTrait<OpTrait::DotLike>()) {
            // Stop tracing when hitting a dot.
            tracedDots.insert(defOp);
          } else
            dfs(defOp);
        }
      }
    };
    dfs(secondDot);
    if (tracedDots.size() != 1)
      return false;

    llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
        loadOpToIndLevelAndUse =
            mlir::triton::loadOpsToIndirectionLevelAndUse(forOp);
    LLVM_DEBUG({
      LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
      for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
        LDBG("  - load: " << *l);
        LDBG("    at indirection level: " << i);
        LDBG("    used by op: " << *u);
      }
    });
    if (loadOpToIndLevelAndUse.empty())
      return false;

    for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
      if (dist != 0)
        return false;
    }

    keyOps.push_back(loads[0]); // FIXME
    keyOps.push_back(loads[1]);
    keyOps.push_back(dots[0]);
    keyOps.push_back(secondDot);
    return true;
  }

  tt::CoarseSchedule::Cluster
  getFAFirstDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                        int numStages) {
    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the first dot to its own stage (numStages - 2), the
    // rest of the computation will be in stage (numStages - 1). The two loads
    // will be in stage 0 and 1.
    SmallVector<Operation *> keyOps;
    if (!isFlashAttention(forOp, keyOps)) {
      LDBG("isFlashAttention returns false");
      return schedule.clusters.begin();
    }
    // firstLoad: keyOps[0]
    tt::CoarseSchedule::Cluster rootUsersCluster =
        schedule.clusters.newAtFront();
    tt::CoarseSchedule::Cluster loadCluster = schedule.clusters.newAtBack();
    schedule.insert(keyOps[0], 0, loadCluster);
    schedule.insert(keyOps[1], 1, loadCluster);
    schedule.insert(keyOps[2], numStages - 2, rootUsersCluster);
    schedule.insert(keyOps[3], numStages - 1, rootUsersCluster);
    return schedule.clusters.begin();
  }

  tt::CoarseSchedule::Cluster
  getFASecondDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                         int numStages) {
    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the second dot to its own stage (numStages - 1), the
    // rest of the computation will be in stage (numStages - 2). The two loads
    // will be in stage 0 and 1.
    return schedule.clusters.begin();
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;
    for (scf::ForOp forOp : loops) {
      int loopNumStages = getNumStagesOrDefault(forOp);
      tt::CoarseSchedule coarseSchedule(loopNumStages);
      tt::CoarseSchedule::Cluster afterPrologue;

      std::string loopSchedule = getLoopScheduleOrDefault(forOp);
      if (loopSchedule == "default") {
        afterPrologue =
            getDefaultLoopSchedule(forOp, coarseSchedule, loopNumStages);
      } else if (loopSchedule == "FA_firstDot") {
        afterPrologue =
            getFAFirstDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else if (loopSchedule == "FA_secondDot") {
        afterPrologue =
            getFASecondDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else {
        assert(false && "unrecognized loop schedule");
      }
      // Go through schedule and assign (stage, cluster).
      // shift so afterPrologue will be at clusterId 0
      auto ctx = forOp.getContext();
      for (auto [op, stage_, cluster] : coarseSchedule.getOpsInOrder(forOp)) {
        op->setAttr("loop.stage",
                    IntegerAttr::get(IntegerType::get(ctx, 32), stage_));
        op->setAttr("loop.cluster",
                    IntegerAttr::get(IntegerType::get(ctx, 32),
                                     *cluster - *afterPrologue));
        LLVM_DEBUG({
          LDBG("set stage " << stage_ << " cluster " << (*cluster));
          op->dump();
        });
      }
    }
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
