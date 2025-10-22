from __future__ import annotations

import typing

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_scatter import scatter
from rich import print


class MoE(nn.Module):
	def __init__(
			self,
			input_dim: int,
			num_experts: int = 4,
			expert_hidden_dim: int = 128,
			expert_output_dim: int = 128,
			experts_num_layers: int = 2,
			gating_input_dim: int = 128,
			gating_hidden_dim: int = 32,
			dtype: torch.dtype = torch.float32,
	) -> None:
		super().__init__()

		self.num_experts: int = num_experts

		self.experts: nn.ModuleList = nn.ModuleList([
			gnn.MLP(
					in_channels=input_dim,
					hidden_channels=expert_hidden_dim,
					out_channels=expert_output_dim,
					num_layers=experts_num_layers,
					dtype=dtype,
			)
			for _ in range(num_experts)
		])

		self.gating_network: nn.Module = nn.Sequential(
				nn.Linear(
						gating_input_dim,
						gating_hidden_dim,
						dtype=dtype
				),
				nn.ReLU(),
				nn.Linear(
						gating_hidden_dim,
						num_experts,
						dtype=dtype
				),
				nn.Softmax(dim=-1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Compute expert outputs
		expert_outputs = torch.stack([expert(x) for expert in self.experts],
		                             dim=1)  # Shape: (batch_size, num_experts, expert_output_dim)

		# Compute gating weights
		gating_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)
		gating_weights = gating_weights.unsqueeze(-1)  # Shape: (batch_size, num_experts, 1)

		# Weighted sum of expert outputs
		output = torch.sum(gating_weights * expert_outputs, dim=1)  # Shape: (batch_size, expert_output_dim)

		return output


class UnParallelMoE(nn.Module):
	def __init__(
			self,
			input_dim: int,
			num_experts: int = 4,
			expert_hidden_dim: int = 128,
			expert_output_dim: int = 128,
			experts_num_layers: int = 2,
			gating_input_dim: int = 128,
			gating_hidden_dim: int = 32,
			expert_activation_threshold: float = 0.5,
			dtype: torch.dtype = torch.float32,
	) -> None:
		super().__init__()

		self.num_experts: int = num_experts
		self.expert_activation_threshold: float = expert_activation_threshold

		self.experts: nn.ModuleList = nn.ModuleList([
			gnn.MLP(
					in_channels=input_dim,
					hidden_channels=expert_hidden_dim,
					out_channels=expert_output_dim,
					num_layers=experts_num_layers,
					dtype=dtype,
			)
			for _ in range(num_experts)
		])

		self.gating_network: nn.Module = nn.Sequential(
				nn.Linear(
						gating_input_dim,
						gating_hidden_dim,
						dtype=dtype
				),
				nn.ReLU(),
				nn.Linear(
						gating_hidden_dim,
						num_experts,
						dtype=dtype
				),
				nn.Softmax(dim=-1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Compute gating weights
		gating_weights = self.gating_network(x)  # Shape: (batch_size, num_experts)

		# Activate only experts with weights above the threshold
		active_experts = (gating_weights > self.expert_activation_threshold).float()  # Shape: (batch_size, num_experts)
		gating_weights = gating_weights * active_experts
		gating_weights = gating_weights / (gating_weights.sum(dim=-1, keepdim=True) + 1e-9)  # Re-normalize
		gating_weights = gating_weights.unsqueeze(-1)  # Shape: (batch_size, num_experts, 1)

		# Compute expert outputs
		expert_outputs = torch.stack([expert(x) for expert in self.experts],
		                             dim=1)  # Shape: (batch_size, num_experts, expert_output_dim)

		# Weighted sum of expert outputs
		output = torch.sum(gating_weights * expert_outputs, dim=1)  # Shape: (batch_size, expert_output_dim)

		return output


class AccumulativeMixtureOfExperts(nn.Module):
	def __init__(
			self,
			input_dim: int,
			num_experts: int = 4,
			expert_hidden_dim: int = 128,
			expert_output_dim: int = 128,
			experts_num_layers: int = 2,
			mode: typing.Union["sum", "mean", "stack"] = "stack",
			distance_domain: float = 20,
			dtype: torch.dtype = torch.float32,
	) -> None:
		super().__init__()

		self.num_experts: int = num_experts
		self.mode: str = mode
		self.distance_domains: torch.Tensor = torch.linspace(0.0, distance_domain, num_experts + 1)[1:]
		self.experts: nn.ModuleList = nn.ModuleList([
			gnn.MLP(
					in_channels=input_dim,
					hidden_channels=expert_hidden_dim,
					out_channels=expert_output_dim,
					num_layers=experts_num_layers,
					dtype=dtype,
			) for _ in range(num_experts)
		])

		if mode == "stack":
			self.stack_head = gnn.MLP(
					in_channels=expert_output_dim * num_experts,
					hidden_channels=expert_hidden_dim,
					out_channels=expert_output_dim,
					num_layers=experts_num_layers,
					dtype=dtype,
			)

	def forward(self, data, distances):

		experts_output: typing.List[torch.Tensor] = []

		for expert, domain in zip(self.experts, self.distance_domains):
			temp: torch.Tensor = torch.zeros_like(data)
			temp[distances >= domain] = expert(
					data[distances >= domain]
			)  # Later experts are getting previous data apart from the new data

			experts_output.append(temp)

		if self.mode == "sum":
			output: torch.Tensor = torch.stack(experts_output, dim=0).sum(dim=0)

		elif self.mode == "mean":
			output: torch.Tensor = scatter(
					torch.stack(experts_output, dim=1),
					data[:, -1],
			)
		elif self.mode == "stack":
			output: torch.Tensor = self.stack_head(
					torch.cat(
							experts_output,
							dim=1),
			)


		else:
			raise ValueError(f"Unknown mode: {self.mode}")
		return output
