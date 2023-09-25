![](https://www.banana.dev/lib_zOkYpJoyYVcAamDf/x2p804nk9qvjb1vg.svg?w=340 "Banana.dev")

# Banana.dev clip-interrogator starter template

This is a clip-interrogator starter template from [Banana.dev](https://www.banana.dev) that allows on-demand serverless GPU inference.

You can fork this repository and deploy it on Banana as is, or customize it based on your own needs.

# Running this app

## Deploying on Banana.dev

1. [Fork this](https://github.com/bananaml/demo-clip-interrogator/fork) repository to your own Github account.
2. Connect your Github account on Banana.
3. [Create a new model](https://app.banana.dev/deploy) on Banana from the forked Github repository.

## Running after deploying

1. Wait for the model to build after creating it.
2. Make an API request using one of the provided snippets in your Banana dashboard. However, instead of sending a prompt as provided in the snippet, send the base64 string of the image where you want to apply CLIP to in the following structure:

```python
inputs = {
    "image": "image_base64_string",
}
```

For more info, check out the [Banana.dev docs](https://docs.banana.dev/banana-docs/).

