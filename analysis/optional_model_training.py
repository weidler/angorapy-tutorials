import angorapy as ap

task = ap.make_task("Pendulum-v1")
build_models = ap.get_model_builder(model="simple", model_type="ffn", shared=False)

agent = ap.Agent(build_models, task, horizon=512, workers=12)
agent.drill(n=20, epochs=5, batch_size=256)

if agent.is_root:
    investigator = ap.analysis.Investigator.from_agent(agent)
    env = ap.make_task(agent.env.spec.id, postprocessors=agent.env.postprocessors, render_mode="human")

    for i in range(10):
        investigator.render_episode(env)

agent.save_agent_state()