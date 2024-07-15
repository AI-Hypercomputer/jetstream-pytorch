import unittest
import torch
from jetstream_pt.model_base import ModuleBase


class TestModuleBase(unittest.TestCase):

  def test_get_hf_names_to_real_name(self):

    class MyModule(ModuleBase):

      def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 30)
        self.hf_name("linear1", "model.my_linear1")
        self.hf_name("linear2", "model.my_linear2")
        self.param = torch.nn.Parameter(torch.randn(10))
        self.hf_name("param", "model.param")

    module = MyModule()
    expected_mapping = {
        "model.my_linear1.weight": "linear1.weight",
        "model.my_linear1.bias": "linear1.bias",
        "model.my_linear2.weight": "linear2.weight",
        "model.my_linear2.bias": "linear2.bias",
        "model.param": "param",
    }

    self.assertEqual(module.get_hf_names_to_real_name(), expected_mapping)

  def test_get_sharding_annotations(self):
    class MyModule(ModuleBase):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
        self.embedding = torch.nn.Embedding(100, 50)
        self.inner = InnerModule()

    class InnerModule(ModuleBase):

      def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(50, 100)

    module = MyModule()
    module.annotate_sharding("linear.weight", 0)
    module.annotate_sharding("embedding.weight", 1)
    module.inner.annotate_sharding("fc.weight", 2)

    expected_mapping = {
        "linear.weight": 0,
        "embedding.weight": 1,
        "inner.fc.weight": 2,
    }
    self.assertEqual(module.get_sharding_annotations(), expected_mapping)


if __name__ == "__main__":
  unittest.main()
