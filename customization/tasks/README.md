# Designing Custom Anthropomorphic Tasks

In this tutorial we will show how to design custom _anthropomorphic_ task environments in AngoraPy. We will use AngoraPy's builtin API that extends the common [Gym](https://github.com/openai/gym) API.

### What is an anthropomorphic task?
When building a goal-driven model of a sensorimotor function, we need to define a task that emulates the sensorimotor context the human brain is situated in. To this end, we need to define a task that is _anthropomorphic_, i.e. a task in which the agent controls a motor plant similar to the human body (or, more feasibly, a part of it) and receives inputs from human-like sensory readings. 

## The Anthropomorphic Task API
AngoraPy's anthropomorphic task API builds on [Gym](https://github.com/openai/gym) and implements its Env interface. It is thus lightweight but provides a range of helpful features for designing anthropomorphic tasks. Specifically, it provides an interface to bodies modeled in MuJoCo 

### Interchangeable and Parameterizable Reward Functions


## Thumbs Up: A Custom Dexterous Task
In this section we will step-by-step design a custom task that requires the agent to control a dexterous hand to perform a thumbs up gesture. 

