#pragma once

// NOTE: You might ask: if we make `F::SomeFuncOptions` the same class as `torch::nn::SomeOptions`
// for most functionals, what happens if the user erroneously passes `torch::nn::SomeOptions`
// in their call to functionals, and we later on make `F::SomeFuncOptions` a different class from
// `torch::nn::SomeOptions` which is going to break their code?
//
// Well, the answer is that they will get compile error at that time, which is enough motivation
// for them to look at the documentation again and fix their usage error. Note that we choose not
// to add mechanism to detect this misuse, because it is a valid and efficient use case for some
// of our internal implementation (e.g. passing module options as functional options in a module's
// `forward` method).

#define TORCH_NN_FUNCTIONAL_USE_MODULE_OPTIONS(module_name, functional_options_name) \
namespace functional { \
using functional_options_name = module_name##Options; \
}
