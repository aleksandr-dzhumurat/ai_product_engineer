**Async in FastAPI:** FastAPI runs on Uvicorn (an ASGI server). Route handlers defined as `async def` run directly on the event loop. Handlers defined as plain `def` are automatically offloaded to a thread pool by Starlette (FastAPI's base), so both work — but `async def` is preferred for I/O-bound endpoints to avoid thread overhead.

**Dependency Injection (DI):** FastAPI's `Depends()` mechanism lets you declare reusable components (database sessions, auth checks, config loaders) as dependencies that are automatically resolved and injected into route handlers.

```
Conceptual flow:

@app.get("/predict")
async def predict(model: Model = Depends(get_model), data: InputSchema = Body(...)):
    prediction = model.predict(data)
    return {"result": prediction}

get_model() is called once (or cached), and its return value is injected.

```

**Why this matters for ML serving:** You can inject a singleton model instance, a feature-store client, or a database connection pool — all resolved once at startup and shared across requests. This avoids re-loading heavy models on every request.

# Refs

*   [https://www.linkedin.com/posts/shantanuladhwe_6-interview-question-and-answers-of-fastapi-activity-7405958412024459264-bUtK](https://www.linkedin.com/posts/shantanuladhwe_6-interview-question-and-answers-of-fastapi-activity-7405958412024459264-bUtK?utm_source=share&utm_medium=member_ios&rcm=ACoAABHcLTkB9ZRrPOB4NW-jmLGXwC1oz0SS_hY)
* [FastAPI to deploy ML models](https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea) + [RealPythonTutorial](https://realpython.com/fastapi-python-web-apis/#create-a-first-api)
* [FastAPI project generator](https://github.com/tiangolo/full-stack-fastapi-postgresql)
*  [Streamlit Auth](https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/) , [FastAPI OAuth](https://fastapi.tiangolo.com/tutorial/security/first-steps/)
* [Python service with OAuth](https://www.grizzlypeaksoftware.com/articles?id=5SCpQMgookgKNtupzNHg9K) + [Microservice with authorization](https://betterprogramming.pub/build-a-todo-app-using-a-microservices-architecture-and-use-auth-service-to-protect-its-routes-f8f0d2ad6669) + [Auth schemes](https://blog.restcase.com/4-most-used-rest-api-authentication-methods/) + [Auth headers](https://reqbin.com/req/python/5k564bhv/get-request-bearer-token-authorization-header-example) + [JWT Setup](https://realpython.com/token-based-authentication-with-flask/)
* [How to build Bearer Auth](https://martinlasek.medium.com/tutorial-how-to-build-bearer-auth-8ae3f80b9522)
* [ToDo app with Auth](https://betterprogramming.pub/build-a-todo-app-using-a-microservices-architecture-and-use-auth-service-to-protect-its-routes-f8f0d2ad6669)
* [Знакомство с FastAPI](https://habr.com/ru/post/488468/)
* [FastAPI concurrency](https://www.linkedin.com/feed/update/ugcPost:7235138518937677824)
[step-in-to-ci-cd-a-hands-on-guide-to-building-ci-cd-pipeline-with-github-actions](https://medium.com/@pathirage/step-in-to-ci-cd-a-hands-on-guide-to-building-ci-cd-pipeline-with-github-actions-7490d6f7d8ff)
* [langgraph-fastapi-and-streamlit-gradio-the-perfect-trio-for-ai-development](https://levelup.gitconnected.com/langgraph-fastapi-and-streamlit-gradio-the-perfect-trio-for-ai-development-f1a82775496a)
* [how-to-optimize-fastapi-for-ml-model-serving](https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d)
* [serving html fastapi](https://www.fastapitutorial.com/blog/serving-html-fastapi/)
* [dynamic form values with jinja and fastapi](https://pype.dev/dynamic-form-values-with-jinja-and-fastapi/)
* [fastapi tutorial encoder](https://fastapi.tiangolo.com/tutorial/encoder/)
* [how-dependency-injection-makes-your-fastapi-code-better](https://www.linkedin.com/pulse/how-dependency-injection-makes-your-fastapi-code-better-bob-belderbos-umyze/)