import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs,
                 actor_hidden_dims=None, critic_hidden_dims=None):
        super().__init__()
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 128]

        # Build critic
        critic_layers = []
        critic_in = num_obs + num_privileged_obs
        for h in critic_hidden_dims:
            critic_layers.append(torch.nn.Linear(critic_in, h))
            critic_layers.append(torch.nn.ELU())
            critic_in = h
        critic_layers.append(torch.nn.Linear(critic_in, 1))
        self.critic = torch.nn.Sequential(*critic_layers)

        # Build actor
        actor_layers = []
        actor_in = num_obs
        for h in actor_hidden_dims:
            actor_layers.append(torch.nn.Linear(actor_in, h))
            actor_layers.append(torch.nn.ELU())
            actor_in = h
        actor_layers.append(torch.nn.Linear(actor_in, num_act))
        self.actor = torch.nn.Sequential(*actor_layers)
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
