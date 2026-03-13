This repository is used to run the benchmark on any OpenAI compatible provider.
As a model runner there is OpenAI API, which can set any base url to use different providers.

Benchmark divide into 2 parts: data generation; data evaluation.

***How to run data generation?***

```
python generate_benchmark.py \
  --out-dir benchmark \
  --language en \
  --num-samples 2 \
  --num-sessions 5 \
  --num-events 12 \
  --qa-per-sample 10 \
  --model openai/gpt-5.4
```

***How to run eval?***
* Copy ```env.example``` into ```.env``` and set up any neccessary env variables.
* Run ```task_eval/evaluate_qa.py```. 
Example: ```python task_eval/evaluate_qa.py --data-file=data/locomo10.json 
--out-file=outputs/locomo10_qa.json --model=gpt-4-turbo --batch-size=20``` 

There may be bugs when running, especially if environment variables are used that are not declared in the code.
Currently, repo is still under refactoring. 

For any references see the readme from the benchmark authors ```static/README_ORIGINAL.MD```  

Next steps:
* Run data generation 
* Get rid of any other providers, libs (e.g. gemini, anthropic)
* Make clearer code 
* Add LLM-as-a-judge as an additional perfomance metric
* Add an interface to support various memory mechanisms (not only RAG or model context)