#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers,
                     unsigned requestedRegisters);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numWarpGroups <= 1)
      return;

    // Partition key ops into multiple async tasks.
    doTaskPartition(funcOp, numWarpGroups);
    // Propagate taskId.
    int retCode = doTaskIdPropagate(funcOp);
    if (retCode == -1)
      signalPassFailure();

    // Partition ops into parallel sub ops.
    if (!doDataPartition(funcOp, numWarpGroups - 1))
      signalPassFailure();
    doCodePartition(funcOp, 2, 0); // FIXME
    doTokenLowering(funcOp, numWarpGroups - 1);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
