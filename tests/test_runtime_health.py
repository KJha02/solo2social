from agent import (
    LLMAgent,
    Rung2Agent,
    completion_preview,
    is_degenerate_completion,
    is_degenerate_token_ids,
    parse_action,
    parse_rung2_action,
    render_experiment_prompt,
    terminal_final_line,
)
from env import BanditEnvironment, InfiniteBanditEnvironment, Pull
from main import ROOT, configure_experiment, configure_model, load_config
from rung3 import (
    Rung3Agent,
    Rung3Environment,
    balanced_initial_skills,
    make_observation,
    parse_schedule,
)


def test_action_parsers_require_a_terminal_final_action():
    assert parse_action("reason\nFINAL: PULL 7").value == 7
    assert parse_action("FINAL: PULL 7\ntrailing") is None
    assert parse_rung2_action("FINAL: INNOVATE 0.25").value == 0.25
    assert parse_rung2_action("FINAL: OBSERVE 3").kind == "observe"
    assert terminal_final_line("reasoning\nFINAL: PULL 2") == "FINAL: PULL 2"


def test_degenerate_completion_detection_and_bounded_preview():
    assert is_degenerate_completion("!" * 4096)
    assert is_degenerate_token_ids([123] * 4096)
    assert not is_degenerate_token_ids([1, 2, 3])
    assert not is_degenerate_completion("FINAL: PULL 0")
    assert len(completion_preview("x" * 1000)) == 257


def test_rung3_schedule_parser_and_scorer():
    jobs = [
        {"id": 0, "deadline": 1, "profit": 100},
        {"id": 1, "deadline": 2, "profit": 80},
        {"id": 2, "deadline": 2, "profit": 20},
    ]
    assert parse_schedule("reason\nFINAL: SCHEDULE 0, 1") == [0, 1]
    assert parse_schedule("FINAL: SCHEDULE 0, 1\ntrailing") is None
    assert Rung3Environment.score_schedule(jobs, [0, 1])["reward"] == 1.0
    assert 0 < Rung3Environment.score_schedule(jobs, [0])["reward"] < 1
    assert not Rung3Environment.score_schedule(jobs, [0, 0])["valid"]
    assert not Rung3Environment.score_schedule(jobs, [1, 0])["valid"]
    assert not Rung3Environment.score_schedule(jobs, [99])["valid"]


def test_rung3_initial_skills_are_balanced_and_deterministic():
    first = balanced_initial_skills(seed=7, num_agents=100, num_skills=8)
    second = balanced_initial_skills(seed=7, num_agents=100, num_skills=8)
    assert first == second
    counts = [first.count(skill_id) for skill_id in range(8)]
    assert max(counts) - min(counts) <= 1
    assert all(count > 0 for count in counts)


def test_rung3_social_information_is_cumulative_and_redacted():
    observer = Rung3Agent(0, 0)
    target = Rung3Agent(1, 1)
    target.add_pull(Pull(round=0, agent_id=1, arm_id=1, reward=0.75))
    stats = {
        1: {
            "description": "neutral description",
            "global_use_count": 4,
            "adoption_count": 1,
            "holder_count": 3,
            "global_mean_reward": 0.5,
            "global_reward_count": 4,
            "skill_body": "full body",
        }
    }
    identity = make_observation(
        observer=observer,
        target=target,
        latest_pull=target.pulls[-1],
        stats=stats,
        round_index=1,
        social_info="id",
    ).visible_dict()
    assert "latest_reward" not in identity
    assert "description" not in identity
    payoff = make_observation(
        observer=observer,
        target=target,
        latest_pull=target.pulls[-1],
        stats=stats,
        round_index=1,
        social_info="payoff",
    ).visible_dict()
    assert payoff["latest_reward"] == 0.75
    assert "description" not in payoff
    full = make_observation(
        observer=observer,
        target=target,
        latest_pull=target.pulls[-1],
        stats=stats,
        round_index=1,
        social_info="full",
    ).visible_dict()
    assert full["global_mean_reward"] == 0.5
    assert full["target_reward_history"] == (0.75,)
    assert full["skill_body"] == "full body"


def test_rung3_candidate_is_copied_only_on_first_pull():
    agent = Rung3Agent(0, 0)
    observation = make_observation(
        observer=agent,
        target=Rung3Agent(1, 1),
        latest_pull=Pull(round=0, agent_id=1, arm_id=1, reward=0.4),
        stats={
            1: {
                "description": "description",
                "global_use_count": 1,
                "adoption_count": 0,
                "holder_count": 1,
                "global_mean_reward": 0.4,
                "global_reward_count": 1,
                "skill_body": None,
            }
        },
        round_index=1,
        social_info="payoff",
    )
    agent.add_observation(round_index=1, target_id=1, observation=observation)
    assert 1 not in agent.owned_skill_ids
    copied, cue = agent.adopt(1, round_index=2)
    assert copied and cue["source_observation_id"] == observation.observation_id
    assert not agent.adopt(1, round_index=3)[0]


def test_rung3_skill_oracle_accepts_model_overrides():
    config = load_config(ROOT / "configs/rung3_skill_oracle.json")
    resolved = configure_model(
        config,
        model_name="Qwen/Qwen3-8B",
        model_label="qwen3_8b",
        tensor_parallel_size=1,
    )
    assert resolved["condition"] == "rung3_skill_oracle__qwen3_8b"
    assert resolved["model"]["tensor_parallel_size"] == 1


def test_solo_states_do_not_expose_social_scaffolding():
    rung1 = LLMAgent(0).render_state(
        round_index=0, num_agents=10, num_arms=25, social_enabled=False
    )
    rung2 = Rung2Agent(0).render_rung2_state(
        round_index=0,
        num_agents=10,
        landscape="unstructured",
        coordinate_scale=10**9,
        coordinate_innovation=True,
        social_enabled=False,
    )
    assert "observation target" not in rung1.lower()
    assert "social observation" not in rung1.lower()
    assert "observation target" not in rung2.lower()


def test_v3_reward_shape_and_regimes_are_deterministic_and_matched():
    static = BanditEnvironment(
        seed=7,
        num_agents=10,
        num_arms=25,
        arm_mean_scale=1.0,
        reward_noise_std=1.0,
    )
    changing = BanditEnvironment(
        seed=7,
        num_agents=10,
        num_arms=25,
        arm_mean_scale=1.0,
        reward_noise_std=1.0,
        regime_period=25,
    )
    assert (static.arm_means == changing.arm_means).all()
    first = changing.arm_means.copy()
    changing.begin_round(25)
    assert not (first == changing.arm_means).all()
    heavy = InfiniteBanditEnvironment(
        seed=7,
        num_agents=10,
        landscape="unstructured",
        arm_mean_scale=1.0,
        reward_noise_std=1.0,
        reward_shape=0.6,
        coordinate_innovation=True,
    )
    assert heavy.innovate(
        agent_id=0, innovation_index=0, round_index=0, coordinate=0.25
    ).arm_id == 250_000_000


def test_v3_overrides_create_explicit_condition_ids():
    config = load_config(ROOT / "configs/rung2_structured_social_action_payoff.json")
    resolved = configure_experiment(
        config,
        output_root="runs/v3/rung2",
        prompt="prompts/v3_rung2_social.txt",
        condition_prefix="v3_",
        num_agents=10,
        num_arms=None,
        budget_mode_override="use_it_or_lose_it",
        guidance="strategic",
        reward_shape=0.6,
        spatial_length_scale=0.05,
        regime_period=25,
        coordinate_innovation=True,
    )
    assert resolved["environment"]["num_agents"] == 10
    assert "strategic" in resolved["condition"]
    assert "shape_0p6" in resolved["condition"]
    assert "corr_0p05" in resolved["condition"]
    assert "regime_25" in resolved["condition"]
    prompt = render_experiment_prompt(
        (ROOT / resolved["prompt"]).read_text(encoding="utf-8"), resolved
    )
    assert "{{" not in prompt
    assert "Nearby coordinates are weakly informative" in prompt
