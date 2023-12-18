#ifndef XLA_TORCH_XLA_CSRC_OPS_TOKEN_ATTENTION_H_
#define XLA_TORCH_XLA_CSRC_OPS_TOKEN_ATTENTION_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class TokenAttention : public XlaNode {
 public:
  TokenAttention(const torch::lazy::Value& q,
                 const torch::lazy::Value& k,
                 const torch::lazy::Value& v,
                 const torch::lazy::Value& kv_read_indices,
                 std::vector<int64_t> seq_len,
                 double scale,
                 bool prefill);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;
  
  const std::vector<int64_t>& seq_lens() { return seq_lens_; }

  double scale() const { return scale_; }
  
  bool prefill() const { return prefill_; }
  
 private:
  std::vector<int64_t> seq_lens_;
  double scale_;
  bool prefill_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_TOKEN_ATTENTION_H_
