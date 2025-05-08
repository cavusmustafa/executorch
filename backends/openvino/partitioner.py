# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from typing import Callable, final, List, Optional, Tuple

import torch
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from openvino.frontend.pytorch.torchdynamo.op_support import (  # type: ignore[import-untyped]
    OperatorSupport,
)

from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase


class OpenvinoOperatorsSupport(OperatorSupportBase):

    def __init__(
        self,
        op_types_to_skip: Optional[set] = None,
        op_names_to_skip: Optional[set] = None,
    ) -> None:
        """
        Initializes the OpenvinoOperatorsSupport class.

        :param op_types_to_skip: A set of operator types to skip during support checking.
        :param op_names_to_skip: A set of operator names to skip during support checking.
        """
        if op_types_to_skip is None:
            op_types_to_skip = set()
        if op_names_to_skip is None:
            op_names_to_skip = set()

        self._op_types_to_skip = op_types_to_skip
        self._op_names_to_skip = op_names_to_skip

    def is_node_supported(self, _, node: torch.fx.Node) -> bool:
        """
        Checks if a given node is supported by OpenVINO.

        :param node: The FX graph node representing an operation.
        :return: True if the node is supported, otherwise False.
        """
        if node.op != "call_function":
            return False

        options: list[str] = []
        if not isinstance(node.target, str):
            op_type = node.target.__name__
        else:
            op_type = str(node.target)

        if op_type in self._op_types_to_skip or node.name in self._op_names_to_skip:
            print(
                f"[OpenVINO Backend] The {op_type} operator with name '{node.name}' is skipped."
            )
            return False

        supported_ops = OperatorSupport(options)._support_dict
        if op_type == "getitem":
            return True

        if "torch.ops." + str(op_type) in supported_ops:
            return True
        else:
            print("Op not supported: ", "torch.ops." + str(op_type))


        return False


@final
class OpenvinoPartitioner(Partitioner):

    def __init__(
        self,
        compile_spec: List[CompileSpec],
        op_types_to_skip: Optional[set] = None,
        op_names_to_skip: Optional[set] = None,
    ) -> None:
        """
        Initializes the OpenvinoPartitioner class.

        :param compile_spec: A list of compile specifications for OpenVINO.
        :param op_types_to_skip: A set of operator types to skip during partitioning.
        :param op_names_to_skip: A set of operator names to skip during partitioning.
        """
        self.delegation_spec = DelegationSpec(OpenvinoBackend.__name__, compile_spec)
        self._op_types_to_skip = op_types_to_skip
        self._op_names_to_skip = op_names_to_skip

    def ops_to_not_decompose(
        self,
        ep: ExportedProgram,
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        """
        Returns a tuple containing a list of operations that should not be decomposed
        and an optional function to filter nodes.

        :param ep: The exported program.
        :return: A tuple consisting of a list of ops to keep and an optional filtering function.
        """
        ops_not_decompose = [
            torch.ops.aten.pixel_shuffle.default,
            torch.ops.aten.upsample_bilinear2d.default,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.upsample_nearest2d.default,
            torch.ops.aten.upsample_nearest2d.vec,
        ]
        return (ops_not_decompose, None)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        """
        Partitions an exported program into supported and unsupported segments.

        :param exported_program: The exported program.
        :return: A PartitionResult containing the partitioned graph and delegation tags.
        """

        self._op_names_to_skip = set()
        print("DEBUG - OpenvinoPartitioner - graph")
        #print(exported_program.graph_module.code)
        for node in exported_program.graph_module.graph.nodes:
            if str(node.op).strip() == "call_function" and str(node.target.__name__).strip() == "aten.slice_copy.Tensor":
            #if str(node.op).strip() == "call_function" and str(node.target.__name__).strip() == "aten.slice_copy.Tensor" and str(node.name).strip() == "aten_slice_copy_tensor_6":
                print("\tDEBUG - OpenvinoPartitioner - slice_copy - op: ", node.op, ", target: ", node.target.__name__, ", name: ", node.name)
                if not (len(node.all_input_nodes) == 3):
                    continue
                slice_copy_in0 = node.all_input_nodes[0]
                if not (str(slice_copy_in0.op).strip() == "placeholder"):
                    continue
                print("\t\tDEBUG - OpenvinoPartitioner - slice_copy_in0 - op: ", slice_copy_in0.op, ", target: ", slice_copy_in0.target, ", name: ", slice_copy_in0.name)
                slice_copy_in1 = node.all_input_nodes[1]
                if not (str(slice_copy_in1.op).strip() == "call_function" and str(slice_copy_in1.target.__name__).strip() == "_local_scalar_dense.default"):
                    continue
                print("\t\tDEBUG - OpenvinoPartitioner - slice_copy_in1 - op: ", slice_copy_in1.op, ", target: ", slice_copy_in1.target.__name__, ", name: ", slice_copy_in1.name)
                slice_copy_in2 = node.all_input_nodes[2]
                if not (str(slice_copy_in2.op).strip() == "call_function" and str(slice_copy_in2.target.__name__).strip() == "add"):
                    continue
                print("\t\tDEBUG - OpenvinoPartitioner - slice_copy_in2 - op: ", slice_copy_in2.op, ", target: ", slice_copy_in2.target.__name__, ", name: ", slice_copy_in2.name)
                #for input_node in node.all_input_nodes:
                #    print("\tDEBUG - OpenvinoPartitioner - input_node - op: ", input_node.op, ", target: ", input_node.target, ", name: ", input_node.name)
                self._op_names_to_skip.add(node.name)

        partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            OpenvinoOperatorsSupport(self._op_types_to_skip, self._op_names_to_skip),
            allows_single_node_partition=True,
        )
        partition_list = partitioner.propose_partitions()
        print("\tDEBUG - part - size: ", partition.size())

        partition_tags = {}
        for partition in partition_list:
            print("\tDEBUG - part - size: ", partition.size())
            for node in partition.nodes:
                tag = f"tag{partition.id}"
                node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program, partition_tags=partition_tags
        )
