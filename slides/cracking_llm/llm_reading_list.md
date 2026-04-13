
[![Лекция 03 vol 2. Векторизация текста Word2Vec Transformers](http://img.youtube.com/vi/csqW3HF_3p8/0.jpg)](http://www.youtube.com/watch?v=csqW3HF_3p8 "Лекция 03 vol 2. Векторизация текста Word2Vec Transformers")

1️⃣ Начать можно с практики — продуктовые кейсы от ребят из FinAI (делают ассистента для команды саппорта).
Блог свежий, много интересных применений: в основном RAG, немного про обучение эффективных узкоспециализированных SLM.
Пишут обо всём — от инфры до аналитики продуктовых экспериментов.
👉 fin.ai/research (https://fin.ai/research/)

Дальше идут углубленные материалы

2️⃣ Основа и фундамент — (https://cme295.stanford.edu/syllabus/) свежий курс CME295 от Стэнфорда.
Курс про трансформеры: лекции, видосы, нормальная структура материала.
Для YouTube роликов я использую notebooklm (https://notebooklm.google.com/) — удобно вытаскивать конспекты и делать инфографику.

3️⃣ Вопросы экзамена по CME295 (https://cme295.stanford.edu/exams/midterm.pdf) идеальны для собесов.
Хорошая подборка: всё разбито по пунктам, покрывает практически весь курс. Отличный инструмент для подготовки.

4️⃣ Чтобы приземлить теорию — курс по LLM (https://huggingface.co/learn/llm-course/en/chapter6/1) от HuggingFace, особенно классная часть про токенайзеры + много практики про деплой джобов в HuggingFace Cloud.

* [Top 24 LLM Questions Asked at DeepMind](https://buildml.substack.com/p/top-24-llm-questions-asked-at-deepmind)
* [genai-llm-ml-case-studies](https://github.com/themanojdesai/genai-llm-ml-case-studies/tree/main)
* [llm-system-design](https://www.systemdesignhandbook.com/guides/llm-system-design)
* [ai-system-design-interview-questions](https://www.educative.io/blog/ai-system-design-interview-questions)
* [rag-evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation)
* [the-ultimate-ai-research-engineer-interview-guide](https://www.sundeepteki.org/advice/the-ultimate-ai-research-engineer-interview-guide-cracking-openai-anthropic-google-deepmind-top-ai-labs#:~:text=Distributed%20Training%20Architectures%20The%20standard,of%20a%20single%20Nvidia%20A100%2FH100)
* [generative-ai-system-design-interview](https://igotanoffer.com/en/advice/generative-ai-system-design-interview#:~:text=Example%20questions%3A)



5️⃣ Если хочется побольше инженерных деталей:
* Nebius LLM Engineering (https://github.com/Nebius-Academy/LLM-Engineering-Essentials) — упор на метрики
* LLMOps Essential (https://github.com/Nebius-Academy/LLMOps-Essentials) — полезный материал ( прикольные штуки типа деплоя в Kubernetes)
* [defeating-nondeterminism-in-llm-inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
* Unsloth [fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

6️⃣ Если вкатывани в LLM тяжело идет, начните с базы.
Стэнфорд выложил CS230 — отличный фундаментальный курс по DL
* слайды (https://cs230.stanford.edu/syllabus/)
* видосы (https://www.youtube.com/playlist?list=PLoROMvodv4rNRRGdS0rBbXOUGA0wjdh1X)

---

Материала хватает минимум на пару недель плотного погружения.
Идеально для новогодних каникул 🎄✨


Transformers

* [video explanation](https://youtu.be/ECR4oAwocjs) 
* [blog post](https://poloclub.github.io/transformer-explainer/)


Обучение эмбеддингов на своих внутренних данных

* [почему это важно](https://fin.ai/research/finetuning-retrieval-for-fin/)
* [как обучать](https://arxiv.org/abs/2512.21021)

📚 **Источники:**
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — оригинальный Transformer
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/) — отличная визуализация
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) — Word2Vec
- [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

---


- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — описание всех трёх типов
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — код с комментариями
- [Transformers explained](https://www.linkedin.com/posts/nicole-koenigstein_transformers-the-definitive-guide-activity-7411413196646846466-PZpQ?utm_source=share&utm_medium=member_ios&rcm=ACoAABHcLTkB9ZRrPOB4NW-jmLGXwC1oz0SS_hY)
- [CS25 Transformers intro](https://youtu.be/XfpMkf4rD6E?si=A0ckxe7ZkndQxWEe)
[CS338 NLPFineTuned](https://www.cs.utexas.edu/~gdurrett/courses/sp2021/lectures/lec19-1pp.pdf)
- [NLP interview questions](https://www.linkedin.com/posts/sumanth077_top-50-llm-interview-questions-a-comprehensive-activity-7400863663253028864-2oPM)
- [encoders vs decoders](https://www.linkedin.com/posts/mary-newhauser_not-all-llms-generate-text-most-people-share-7402121282898739201-mOSi/)
- [self-attention-from-scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html?utm_source=substack&utm_medium=email)
- [Self attention](https://youtu.be/Bg8Y5q1OiP0)
- [Transformers cheatsheet](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/blob/main/en/cheatsheet-transformers-large-language-models.pdf)
- [Mixture of experts](https://www.youtube.com/watch?v=CDnkFbW-uEQ)

## LLM datasets

- [amazon-reviews-2023](https://amazon-reviews-2023.github.io/)


## API providers

- [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://instructor-embedding.github.io/)
- (proprietary) https://txt.cohere.ai/multilingual/ + https://storage.googleapis.com/cohere-assets/blog/embeddings/multilingual-embeddings-demo
- [OpenAI](https://openai.com/api/pricing/)
    - [OpenAI cookbook](https://github.com/openai/openai-cookbook/tree/main/examples)
        - [OpenAI cookbook: Question answering using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
        - [OpenAI cookbook: Olympics-1-collect-data](https://github.com/openai/openai-cookbook/blob/57024c70cff473fb520105e9aea3ab4e514be0df/examples/fine-tuned_qa/olympics-1-collect-data.ipynb)
        - [OpenAI developer build hours](https://academy.openai.com/public/collections/developer-build-hours-2025-03-20)
- [Gemini cookbook](https://github.com/google-gemini/cookbook)
- [Gemini prompting strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
- [workshop-build-with-gemini](https://github.com/patrickloeber/workshop-build-with-gemini)
- [Perplexity prompt guide](https://docs.perplexity.ai/guides/prompt-guide)

## Papers


### LLM Courses

* [HuggingFace NLP course](https://huggingface.co/learn/nlp-course)
* [Gemini CLI code and create with an open source agent](https://www.deeplearning.ai/short-courses/gemini-cli-code-and-create-with-an-open-source-agent)
* [Prompt compression and query optimization](https://www.deeplearning.ai/short-courses/prompt-compression-and-query-optimization)
* [CS 865 course](https://people.cs.umass.edu/~miyyer/cs685/schedule.html)
* [Kaggle prompt engineering](https://www.kaggle.com/whitepaper-prompt-engineering)
* [YouTube-Blog/LLMs](https://github.com/ShawhinT/YouTube-Blog/blob/main/LLMs/README.md)
* [Azure few shot prompting](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/leveraging-dynamic-few-shot-prompt-with-azure-openai/ba-p/4225235)
* [Prompt engineering technique](https://www.linkedin.com/feed/update/ugcPost:7208896181089808384)
* [LLM-openAI-MongoDB-python](https://www.linkedin.com/feed/update/activity:7246974245312831488)
* [AI agents for video generation](https://www.linkedin.com/feed/update/activity:7248372860925173760)
* [COHERE COURSE](https://docs.cohere.com/docs/llmu)
* [Promt course](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction)
* [Prompt engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
* [Notebook collection fine tuning](https://github.com/ashishpatel26/LLM-Finetuning)
* [Mistral finetuning notebooks](https://github.com/mistralai/mistral-finetune)
* [LitLLM finetuning](https://www.linkedin.com/feed/update/activity:7245778941909614592)
* [Alpaca + Gemma 2 9b notebook finetuning](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)
* [Github LLM course](https://github.com/mlabonne/llm-course)
* [mlabonne/llm-course](https://github.com/mlabonne/llm-course)
* [HuggingFace open source model](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/)
* [Huggingface zero-shot](https://www.notion.so/Huggingface-zero-shot-1cea1f1c934e80179d46fa90f850c5f8?pvs=21)
* [deploy to HuggingFace Space](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/15/deployment)
* [decodingml/llm-twin-course](https://github.com/decodingml/llm-twin-course)
* [hamzafarooq/advanced-llms-course](https://github.com/hamzafarooq/advanced-llms-course)
* [DataTalksClub/llm-zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp/)
* [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners)
* [finetune using unsloth](https://www.linkedin.com/posts/migueloteropedrido_mlops-machinelearning-datascience-activity-7315652239484772353-9JBD)
* [unsloth fine tuning](https://www.linkedin.com/posts/paoloperrone_fine-tuning-big-llms-used-to-be-a-luxuryslow-activity-7336045384399470597-T7m9)
* [LoRA finerune Unsloth](https://www.linkedin.com/posts/migueloteropedrido_mlops-machinelearning-datascience-activity-7331601051952877568-GO3r)
* [LLamMa factory fine tune](https://github.com/hiyouga/LlamaFactory)
* [fine tuning](https://www.linkedin.com/feed/update/activity:7208093607122145280)



LLMOps

* [AWS infrastructure for GenAI](https://rebirth.devoteam.com/2024/06/20/aws-infrastructure-for-gen-ai/?trk=feed-detail_main-feed-card_feed-article-content)
* [ML deployments](https://www.linkedin.com/posts/aurimas-griciunas_genai-llm-machinelearning-activity-7246429092286283778-iDZ7)
* [arazvant_the-4-patterns-of-llm-inference-deploying-activity](https://www.linkedin.com/posts/arazvant_the-4-patterns-of-llm-inference-deploying-activity-7406332327548665856-ZYTO)
* [run LLM model locally](https://yc.prosetech.com/running-your-very-own-local-llm-6d4db99c0611)
* [weaviate.io/blog](https://weaviate.io/blog)
* [weaviate.io/blog/late-interaction-overview](https://weaviate.io/blog/late-interaction-overview)
* [run LLM locally](https://abishekmuthian.com/how-i-run-llms-locally/)
* [Knowledge graph + LLM](https://blog.selman.org/2024/05/25/knowledge-graphs-question-answering)
* [Text to graph of concepts](https://medium.com/data-science/how-to-convert-any-text-into-a-graph-of-concepts-110844f22a1a)
* [YandexGPT: instruct](https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct)
* [integrating-vector-databases-with-llms-a-hands-on-guide](https://medium.com/@mlengineering/integrating-vector-databases-with-llms-a-hands-on-guide-82d2463114fb)
* [Gemini embeddings](https://www.notion.so/How-to-find-a-job-83abd9eeecd14af19c01d9be573d08f9?pvs=21)
* [OpenAI for speech recognition locally](https://vas3k.club/post/18916/)
* [HuggingFace opensource model](https://learn.deeplearning.ai/courses/open-source-models-hugging-face/lesson/1/introduction)
* [Deploy Large Language Open Source Models](https://medium.com/ai-mind-labs/how-to-deploy-large-open-source-llms-3c62d216383b)
* [Vector DB (linkedin)](https://www.linkedin.com/feed/update/activity:7176150262045155328)
* [Youtube: LLaMA finetuned](https://youtu.be/zHv5pA-lxAA?si=ogtPUuVBZ4njGS1C)
* [fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications)
* [deploy-tiny-llama-on-aws-ec2](https://towardsdatascience.com/deploy-tiny-llama-on-aws-ec2-f3ff312c896d)
* [LLM food delivery](https://github.com/lucastononro/llm-food-delivery)
* [hands-on-llms](https://github.com/iusztinpaul/hands-on-llms)
* [Chatbot with Langchain and Weawite](https://medium.com/@s.rashwand/how-to-build-a-chatbot-smarter-than-chatgpt-quickly-using-langchain-and-weaviate-f6309cc86e09)
* [Fine tuned Customer Service ChatBot](https://medium.com/data-science-at-microsoft/how-to-build-a-fine-tuned-customer-service-chatbot-with-python-and-openai-88e221e5bf36#4713)
* [intent-creation-extraction-using-large-language-models](https://cobusgreyling.medium.com/intent-creation-extraction-using-large-language-models-e4634fb5db98)
* [Fine-tune bert for data extraction](https://www.linkedin.com/posts/mary-newhauser_you-should-be-using-bert-based-slms-because-activity-7397287634982719489-CzqD)
* [QuestionAnswering with Fine-tuned BERT](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626)
* [LLM metrics](https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5)
* [Mistral pretrained](https://www.linkedin.com/feed/update/activity:7177436861995343872)
* [MistralAI fine tuning](https://medium.com/@codersama/fine-tuning-mistral-7b-in-google-colab-with-qlora-complete-guide-60e12d437cca)
* [LLAMA 3 Collab finetuning](https://www.linkedin.com/feed/update/activity:7221621362417700867)
* [LLAMA 3.2 Collab finetuning](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing)
* [LLaMA 1B finetuning](https://towardsdatascience.com/i-fine-tuned-the-tiny-llama-3-2-1b-to-replace-gpt-4o-7ce1e5619f3d)
* [Uncensored LLM](https://www.youtube.com/watch?v=BntGOaMrB90)
* [Open food dataset](https://www.linkedin.com/feed/update/activity:7255614485661380609)
* [[HuggingFace]Uncensored LLM](https://huggingface.co/blog/mlabonne/abliteration)
* [Mistral's 8x7B model and its uncensored](https://youtu.be/GyllRd2E6fg?si=GiuIXmP1eptd8d5t)
* [Uncensored LLM](https://youtu.be/BntGOaMrB90?si=YYhLM6kWO5Kfyt53)
* [Uncensored LLM](https://youtu.be/Jus-0q4hRS8?si=fLydUGHk4XLZ0V5t)
* [Uncensored LLM](https://youtu.be/50x0yDWkHjw?si=5cdRbD8vKOqnpsd1)
* [Fine-tune LLM](https://medium.com/towards-data-science/how-to-efficiently-fine-tune-your-own-open-source-llm-using-novel-techniques-code-provided-03a4e67d1b48)
* [Colab Notebooks fine-tuned LLMs](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7)
* [pytorch.org/blog/torchtune-fine-tune-llms](https://pytorch.org/blog/torchtune-fine-tune-llms)
* [Neptune AI: LLM data preparation](https://nebius.ai/blog/posts/data-preparation/llm-dataprep-techniques?utm_content=188525294&utm_medium=social&utm_source=linkedin&hss_channel=lcp-89802307)
* [Microsoft Auto Gen](https://microsoft.github.io/autogen/0.2/docs/autogen-studio/getting-started/)
* [How claude works](https://www.linkedin.com/posts/leadgenmanthan_how-claude-works-ugcPost-7322536945983610880-Jfeg)
* [smol-course](https://github.com/huggingface/smol-course)
* [llm.clickhouse.com](https://llm.clickhouse.com/c/new)
* [nanovllm_tensor_parallel_kernel_fusion](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html)
* [training-a-tokenizer-for-llama-model](https://machinelearningmastery.com/training-a-tokenizer-for-llama-model/)
* [LLM warm-up](https://www.linkedin.com/feed/update/urn:li:activity:7431349832335687681)

* [axsaucedo/kaos](https://github.com/axsaucedo/kaos)
* [Transformer Lab - LLM platform](https://www.linkedin.com/posts/avi-chawla_i-found-a-100-open-source-toolkit-to-work-ugcPost-7321845186924285953-vbol)
* [DiFy - LLM platform](https://github.com/langgenius/dify)
* [LangFuse, LangFlow](https://www.linkedin.com/posts/stevesuarez21_github-langflow-ailangflow-langflow-is-activity-7332799157109223424-2cvf)
* [dify + ollama](https://docs.dify.ai/development/models-integration/ollama)
* [GGUF ollama](https://www.linkedin.com/posts/arazvant_for-engineers-who-want-to-optimize-llms-for-activity-7399816925830098944-trNv)
* [product quantization](https://www.linkedin.com/posts/maxbuckley_product-quantization-quantizing-the-query-activity-7395201104814292992-yM9l)
* [difyopenrouterk8s-ku](https://www.ifb.me/en/blog/en/ai/difyopenrouterk8s-ku)
* [ollama to kubernetes](https://daegonk.medium.com/deploying-ollama-on-kubernetes-8a79d0192d24)
* [OpenRouter](https://zhurnalus.artlebedev.ru/go/3B2A6EAC-11AD-4BEB-B049-474198037FA5_4BF432B4-BD60-4066-AB10-FA899D03E968/)
* [LLM inference](https://m.youtube.com/watch?v=9tvJ_GYJA-o)
* [LLM inference workload explained](https://www.youtube.com/watch?v=z2M8gKGYws4)
* [LLM inference](https://llm.clickhouse.com/c/new)
* [aurimas-griciunas_genai-llm-machinelearning-activity](https://www.linkedin.com/posts/aurimas-griciunas_genai-llm-machinelearning-activity-7246429092286283778-iDZ7)
* [run pytorch faster](https://www.linkedin.com/feed/update/activity:7240721255853654016)
* [PyTorch debugging](https://machinelearningmastery.com/debugging-pytorch-machine-learning-models-a-step-by-step-guide/)
* [MLOps maturity model with Azure Machine Learning](https://techcommunity.microsoft.com/blog/machinelearningblog/mlops-maturity-model-with-azure-machine-learning/3520625)
* [Real time recommender in production](https://www.linkedin.com/posts/pauliusztin_building-a-production-real-time-personalized-activity-7281255764348784640-Mbo8/)
* [Architect a real time machine learning inference application](https://www.linkedin.com/feed/update/activity:7079758311134257152)
* [Speed Up feature engineering](https://eng.snap.com/speed-up-feature-engineering?trk=feed_main-feed-card_reshare_feed-article-content)
* [ML system design interview](https://www.youtube.com/live/JvrlZIc0ObE?feature=share&t=3344)
* [System design for recommendations](https://eugeneyan.com/writing/system-design-for-discovery/)
* [ML system design: federated learning](https://nilg.ai/202107/ml-system-design-federated-learning/)
* [ML system design](https://www.evidentlyai.com/ml-system-design)
* [ML Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp)
* [ML in DataBricks](https://www.youtube.com/playlist?app=desktop&list=PL_MIDuPM12MOcQQjnLDtWCCCuf1Cv-nWL)


## Applications

### Chatbots

* [как обучать WebGPT](https://habr.com/ru/company/ods/blog/709222/) на хабре (немного RL) Дальше список похожих сервисов
* [Q&A поверх GPT3](https://simonwillison.net/2023/Jan/13/semantic-search-answers/)
* [how-to-make-a-recommender-system-chatbot-with-llms](https://medium.com/@mrmaheshrajput/how-to-make-a-recommender-system-chatbot-with-llms-770c12bbca4a)
* [LoRA explained](https://medium.com/towards-data-science/lora-intuitively-and-exhaustively-explained-e944a6bff46b)
* [LoRa continued pre-training](https://unsloth.ai/blog/contpretraining)
* [LoRA train](https://www.linkedin.com/feed/update/activity:7246570435566309378)
* [LLM fine tuning techiques](https://www.linkedin.com/posts/avi-chawla_5-techniques-to-fine-tune-llms-explained-activity-7332366608595636224-foFN)
* [P-tune](https://www.kaggle.com/code/sdlee94/llm-p-tuning-starter-training-inference)
* [Chat- templates fine tuning](https://huggingface.co/learn/smol-course/unit1/4)
* [Train with LoRA](https://medium.com/towards-data-science/implementing-lora-from-scratch-20f838b046f1)
* [QLoRA:  fine-tune a 33B-parameter LLM on Google Colab in a just few hours](https://www.linkedin.com/feed/update/activity:7067545842995363840)
* [QLoRa](https://yashugupta-gupta11.medium.com/qlora-efficient-finetuning-of-large-language-model-falcon-7b-using-quantized-low-rank-adapters-2df59a7982d5)
* [AI-powered chatbot for finance](https://medium.com/predict/crafting-an-ai-powered-chatbot-for-finance-using-rag-langchain-and-streamlit-4384a8076960)
* [Google Cloud LLama2](https://medium.com/@woyera/how-to-use-llama-2-with-an-api-on-gcp-to-power-your-ai-apps-77a3c79b585)
* [Optimize sentence transformer with HuggingFace optimum](https://www.philschmid.de/optimize-sentence-transformers)
* [Finetuning T5 + Streamlit](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887)
* [Falcon meets Llama](https://www.linkedin.com/feed/update/urn:li:activity:7067841408451104768) + [How to build LLaMa index](https://medium.com/llamaindex-blog/build-a-chatgpt-with-your-private-data-using-llamaindex-and-mongodb-b09850eb154c)
* [mistral-7b-has-been-released](https://www.linkedin.com/posts/younes-belkada-b1a903145_recently-mistral-7b-has-been-released-to-activity-7117593843826339840-aPO2)
* [Google Gemma 7B](https://www.linkedin.com/feed/update/groupPost:152247-7166101067523461121)
* [Google gemini AI studio](https://blog.google/technology/ai/gemini-api-developers-cloud)
* [LLaMA index + Gemini](https://www.llamaindex.ai/blog/llamaindex-gemini-8d7c3b9ea97e)
* [Google Gemma released](https://www.notion.so/English-phone-Interview-preparing-1817678191da47329a9871c901bfb347?pvs=21)
* [FunctionGemma finetune](https://www.linkedin.com/posts/jacek-golebiowski_google-released-functiongemma-a-270m-parameter-activity-7429225866150375424-ZrlH)
* [LLaMA on CPU](https://towardsdatascience.com/running-llama-2-on-cpu-inference-for-document-q-a-3d636037a3d8)
* [Host LLama2 on GPU](https://medium.com/@yuhongsun96/host-a-llama-2-api-on-gpu-for-free-a5311463c183)
* [Promt techniques](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb)
* [Bard API](https://github.com/dsdanielpark/Bard-API?trk=feed_main-feed-card_reshare_feed-article-content)
* [Airbnb text generation](https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3)
* [Airbnb: content categories](https://medium.com/airbnb-engineering/building-airbnb-categories-with-ml-and-human-in-the-loop-e97988e70ebb)
* [Answering POI-Recommendation Questions using Tourism Reviews](https://www.cse.iitd.ac.in/~mausam/papers/cikm21.pdf)
* [Large Language Model (LLM) Primers](https://www.linkedin.com/feed/update/activity:7050304546526416896/)
* [LLM explained](https://www.linkedin.com/feed/update/ugcPost:7105866304707801088)
* [BERT](https://arxiv.org/abs/2004.12832)
* [CodeBERT A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155v4.pdf)
* [pipelines](https://huggingface.co/docs/transformers/v4.50.0/en/main_classes/pipelines#transformers.pipeline.task)
* [ModernBert](https://huggingface.co/answerdotai/ModernBERT-base#training)
* [finetune for classification](https://m.youtube.com/watch?v=5PFXJYme4ik)
* [Bert fine tune](https://www.notion.so/Bert-fine-tune-1cba1f1c934e807f8cfaddc69fe5ea27?pvs=21)
* [Code Documentation Generation](https://paperswithcode.com/task/code-documentation-generation)
* [TourBERT A Natural Language Processing Model for the Travel Industry.pdf](https://drive.google.com/file/d/162eo414TyNVzfy9ZvIBCe7EuxP2zfKF0/view?usp=sharing)
* [BERT vs GPT3](https://ainextlevel.net/bert-vs-gpt-3-a-deep-dive-into-the-battle-of-the-giant-language-models/)
* [Hugging Face LLaMA](https://www.linkedin.com/feed/update/urn:li:activity:7094988929510043648)
* [GPT quantization](https://www.linkedin.com/feed/update/urn:li:activity:7091799498804322304)
* [Fine-tune LLM](https://www.linkedin.com/feed/update/activity:7077532387147960320)
* [GPT: key, value, query](https://towardsdatascience.com/how-gpt-works-a-metaphoric-explanation-of-key-value-query-in-attention-using-a-tale-of-potion-8c66ace1f470)
* [fine-tuning-flan-t5-xxl-with-deepspeed-and-vertex-ai](https://medium.com/google-cloud/fine-tuning-flan-t5-xxl-with-deepspeed-and-vertex-ai-af499daf694d)
* [fine-tune-fine-tuning-t5-for-text-generation](https://medium.com/@xiaohan_63326/fine-tune-fine-tuning-t5-for-text-generation-c51ed54a7941)
* [LLM fine-tuning](https://www.linkedin.com/feed/update/urn:li:activity:7206634598401789953)
* [using-bertopic-to-analyze-qatar-world-cup-twitter-data](https://medium.com/@cd_24/using-bertopic-to-analyze-qatar-world-cup-twitter-data-a5956c4949f1)
* [paragraph-based-transformer-pretraining-for-multi-sentence-inference](https://www.amazon.science/code-and-datasets/paragraph-based-transformer-pretraining-for-multi-sentence-inference)
* [Hackers guide to language models](https://www.linkedin.com/feed/update/activity:7120787285326893056)
* [Mixtral 8x7B githib](https://replicate.com/nateraw/mixtral-8x7b-32kseqlen)
* [Fine-tuning LLaMA3 with OPRO](https://www.linkedin.com/feed/update/activity:7187052127410692096)
* [finetuning for classification](https://m.youtube.com/watch?v=5PFXJYme4ik)
* [small LLMs fine tuning (SLM)](https://arxiv.org/abs/2505.24189v1)
* [Domain-Specific Small Language Models](https://github.com/virtualramblas/Domain-Specific-Small-Language-Models)
* [domain specific llms](https://github.com/virtualramblas/Domain-Specific-Small-Language-Models/)
* [Small models fine-tuning](https://www.linkedin.com/posts/maxbuckley_fine-tuning-for-making-expert-domain-specific-activity-7369093772732682241-bsSg)
* [you-should-be-using-bert-based-slms-because](https://www.linkedin.com/posts/mary-newhauser_you-should-be-using-bert-based-slms-because-activity-7397287634982719489-CzqD)
* [offline ai assistant](https://medium.com/@korshun.dev/offline-ai-assistant-in-15-mb-ae86786f5ced)
* [Gemma architecture](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)

### GPT

* [Train ChatGPT on your own dataset](https://beebom.com/how-train-ai-chatbot-custom-knowledge-base-chatgpt-api/#:~:text=You%20can%20ask%20further%20questions,kind%20of%20information%20you%20want)
* [Running minimal GPT-Neo_X](https://github.com/zphang/minimal-gpt-neox-20b)
* [GPT-J vs Chat-GPT](https://www.width.ai/post/gpt-j-vs-gpt-3)  + [Advanced NER With GPT-3 and GPT-J](https://towardsdatascience.com/advanced-ner-with-gpt-3-and-gpt-j-ce43dc6cdb9c)
* [Few-Shot NER tuning](https://habr.com/ru/companies/sberbank/articles/649609/)

### LangChain

* [LangChain course](https://learn.deeplearning.ai/langchain-chat-with-your-data)
* [question-answering-service-using-langchain](https://www.anyscale.com/blog/building-a-self-hosted-question-answering-service-using-langchain-ray)
* [langchain-101-build-gptpowered-applications](https://www.kdnuggets.com/2023/04/langchain-101-build-gptpowered-applications.html)
* [preprocess your data. for LLM finetuning](https://www.linkedin.com/feed/update/activity:7071409886906990592/)
* [huggingface-cohere-langflow](https://cobusgreyling.medium.com/huggingface-cohere-langflow-3b43c6c9a859)
* [LangChain Text Summarization](https://www.linkedin.com/posts/damienbenveniste_machinelearning-datascience-artificialintelligence-activity-7071515173471076352-GdYm?utm_source=share&utm_medium=member_desktop)
* [LangChain + FastAPI](https://medium.com/@jaswanth04/streaming-responses-from-llm-using-langchain-fastapi-329f588d3b40)

### Text summarization

* [T5 for text summarization in 7 lines of code](https://medium.com/artificialis/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771)
* [PyTorch: Summarization](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)
* [Train t5 for text Summarization](https://medium.com/askdata/train-t5-for-text-summarization-a1926f52d281)
* [Fine tuning a T5 transformer for any summarization task](https://towardsdatascience.com/fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81)
* [Few-Shot Fine-Tuning SOTA Summarization Models for Medical Dialogues](https://aclanthology.org/2022.naacl-srw.32.pdf)
* [text2text generation flan T5](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/text2text-generation-flan-t5.ipynb)
* [few-shot-learning-with-setfit](https://hutsons-hacks.info/few-shot-learning-with-setfit)

## Hardware

* [Cerebras](https://www.cerebras.ai/)
* [NVIDA competitors](https://www.linkedin.com/posts/emi-andere_i-wrote-deep-dives-on-6-nvidia-competitors-activity-7429975000507449344-DHPE)