#include "torch_xla/csrc/ops/token_attention.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/slicing.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::XlaOp BuildMask(xla::XlaOp score) {
  xla::XlaOp minus_infinity = xla::MinValue(score.builder(), xla::F32);
  const auto& score_dims = ShapeHelper::ShapeOfXlaOp(score).dimensions();
  xla::XlaOp mask = xla::Broadcast(minus_infinity, {1, score_dims[1], score_dims[2]});
  mask = BuildTriu(mask, 1);
  return mask;
}

xla::XlaOp BuildTokenAttention(xla::XlaOp q,
                               xla::XlaOp k,
                               xla::XlaOp v,
                               xla::XlaOp kv_read_indices,
                               const std::vector<int64_t>& seq_lens,
                               double scale,
                               bool prefill) {
  //  num_heads: the number of local attention heads.
  //  input_len: padded total prompt length for all seqs when prefill; or padded num of seqs when decode.
  //   head_dim: attention head dim.
  //  cache_len: KV cache length for all seqs.
  //    seq_len: current seq context length.
  // prompt_len: current seq prompt length.
  
  // q: (num_heads, input_len, head_dim)
  // k: (num_heads, cache_len, head_dim)
  // v: (num_heads, cache_len, head_dim)
  // kv_read_indices: (cache_len,)
  
  std::vector<xla::XlaOp> outputs;
  int64_t start = 0;
  xla::PrimitiveType target_type = XlaHelpers::TypeOfXlaOp(q);
  xla::XlaBuilder* builder = q.builder();
  xla::XlaOp scaling_value = XlaHelpers::ScalarValue<double>(scale, target_type, builder);
  
  for (int i = 0; i < seq_lens.size(); ++i) {
    int64_t seq_len = seq_lens[i];
    
    xla::XlaOp seq_q;
    if (prefill) {
      // (num_heads, prompt_len, head_dim)
      seq_q = xla::SliceInDim(q, start, start + seq_len, 1, 1);
    } else {
      // (num_heads, 1, head_dim)
      seq_q = xla::SliceInDim(q, i, i + 1, 1, 1);
    }
    
    // (num_heads, seq_len, head_dim)
    xla::XlaOp seq_indices = xla::SliceInDim(kv_read_indices, start, start + seq_len, 1, 0);
    xla::XlaOp seq_k = xla::TorchIndexSelect(k, seq_indices, 1, 0);
    
    // (num_heads, prompt_len|1, seq_len)
    xla::XlaOp score = CreateMatMul(seq_q,  xla::Transpose(seq_k, {0, 2, 1})) * scaling_value;
    
    if (prefill) {
      // (num_heads, prompt_len, seq_len)
      xla::XlaOp mask = BuildMask(score);
      score = xla::ConvertElementType(score, xla::F32);
      score = score + mask;
    } else {
      score = xla::ConvertElementType(score, xla::F32);
    }
    
    // (num_heads, prompt_len|1, seq_len)
    score = BuildSoftmax(score, 2);
    score = xla::ConvertElementType(score, target_type);

    // (num_heads, prompt_len|1, head_dim)
    xla::XlaOp seq_v = xla::TorchIndexSelect(v, seq_indices, 1, 0);
    xla::XlaOp output = CreateMatMul(score, seq_v);
    outputs.push_back(output);
    start += seq_len;
  }
  // Stack outputs without padding.
  xla::XlaOp result = xla::ConcatInDim(builder, outputs, 1);
  // Pad as needed.
  int64_t input_len = ShapeHelper::ShapeOfXlaOp(q).dimensions(1);
  int64_t result_len = ShapeHelper::ShapeOfXlaOp(result).dimensions(1);
  // (num_heads, input_len, head_dim)
  if (input_len > result_len) {
    xla::XlaOp pad_value = xla::Zero(builder, target_type);
    result = xla::PadInDim(result, pad_value, 1, 0, input_len - result_len);
  }
  return result;
}

}  // namespace

TokenAttention::TokenAttention(const torch::lazy::Value& q,
                               const torch::lazy::Value& k,
                               const torch::lazy::Value& v,
                               const torch::lazy::Value& kv_read_indices,
                               std::vector<int64_t> seq_lens,
                               double scale,
                               bool prefill)
    : XlaNode(xla_token_attention, {q, k, v, kv_read_indices},
              [&]() { return GetXlaShape(q); },
              /*num_outputs=*/1,
              torch::lazy::MHash(seq_lens.size(), prefill)),
      seq_lens_(std::move(seq_lens)), scale_(scale), prefill_(prefill) {}

torch::lazy::NodePtr TokenAttention::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<TokenAttention>(operands.at(0), operands.at(1),
                                               operands.at(2), operands.at(3),
                                               seq_lens_, scale_, prefill_);
}

XlaOpVector TokenAttention::Lower(LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));
  xla::XlaOp kv_read_indices = loctx->GetOutputOp(operand(3));
  return ReturnOp(BuildTokenAttention(q, k, v, kv_read_indices, seq_lens_, scale_, prefill_), loctx);
}

std::string TokenAttention::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", seq_lens=(" << absl::StrJoin(seq_lens_, ", ")
     << "), scale=" << scale_ << ", prefill=" << prefill_;
  return ss.str();
}

}  // namespace torch_xla
