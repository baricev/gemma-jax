# %% 

def sanitize_text_for_tokenizer(text):
    # Handle basic sanitization
    if not isinstance(text, str):
        try:
            text = text.decode('utf-8')
        except (UnicodeDecodeError, AttributeError):
            text = str(text)
    
    # Remove control characters
    import re
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Handle Unicode normalization
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    return text

# Example texts to sanitize:

raw_text = """
## JIT and caching

With the compilation overhead of the first JIT call, understanding how and when {func}jax.jit caches previous compilations is key to using it effectively.

Suppose we define f = jax.jit(g). When we first invoke f, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of f will reuse the cached code.
This is how jax.jit makes up for the up-front cost of compilation.

If we specify static_argnums, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs.
If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling {func}jax.jit on temporary functions defined inside loops or other Python scopes.
For most cases, JAX will be able to use the compiled, cached function in subsequent calls to {func}jax.jit.
However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined.
This will cause unnecessary compilation each time in the loop:

{code-cell}
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
"""



p1=r"""
Client: BuzzJuice
Brand Information:
Buzzjuice is the only energy drink with all-organic ingredients.
BuzzJuice has created an energy-boosting formula incorporating non-traditional ingredients, such as Ashwagandha, Lion's Mane, L-Theanine, and Yerba mate. The formula produces functional benefits that go beyond just an initial energy boost. The ingredients have a wide-ranging set of additional benefits, such as improved focus, mood enhancement, and immune support.
BuzzJuice comes in 3 unique flavors: Hibiscus Lime, Tamarind Grape, and Lavender Berry.
The packaging consists of only recycled materials and is biodegradable, resulting in zero net-waste.
The target audience is anyone who could benefit from an energy boost and functional focus without sacrificing their health.
Campaign Goals:
Target Audience: ages 14-29, from high-school students all the way to young professionals, and fitness enthusiasts. The BuzzJuice drinker is always on the go, while valuing health, and wellness. They are both health- and trend conscious.
Brand values: Innovation, sustainability, and transparency.
Brand voice and tone: Energetic, motivating, edgy. The tone must resonate with the youthful target audience. Use dynamic language that inspires action. Avoid being overly trendy, and remain authentic.
Deliverables:
Create a concise tagline that communicates the brand's vibe. It should be a short sentence or phrase.

Create a character named Dr. Buzz to be featured in TV and online ads. He should be goofy, and eccentric. Think Doc Brown from the movie Back to the Future. Provide detailed information about wardrobe, hair-and make-up, and voice.

Create an Instagram profile for Dr. Buzz. Provide concepts for the first 2 posts.

Each post must be goofy, and eccentric, like Dr. Buzz himself.

Create 2 TikTok pitches. One should feature Dr. Buzz creating a unique TikTok dance. He must be terrible at dancing, but his energy must be infectious.

Each post must be goofy, and eccentric, like Dr. Buzz himself.
"""


p2=r"""
## Marking arguments as static
If we really need to JIT-compile a function that has a condition on the value of an input, we can tell JAX to help itself to a less abstract tracer for a particular input by specifying static_argnums or static_argnames.
The cost of this is that the resulting jaxpr and compiled artifact depends on the particular value passed, and so JAX will have to re-compile the function for every new value of the specified static input.
It is only a good strategy if the function is guaranteed to see a limited set of static values.

{code-cell}
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))


{code-cell}
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))


To specify such arguments when using jit as a decorator, a common pattern is to use python's {func}functools.partial:

{code-cell}
from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))


## JIT and caching

With the compilation overhead of the first JIT call, understanding how and when {func}jax.jit caches previous compilations is key to using it effectively.

Suppose we define f = jax.jit(g). When we first invoke f, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of f will reuse the cached code.
This is how jax.jit makes up for the up-front cost of compilation.

If we specify static_argnums, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs.
If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling {func}jax.jit on temporary functions defined inside loops or other Python scopes.
For most cases, JAX will be able to use the compiled, cached function in subsequent calls to {func}jax.jit.
However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined.
This will cause unnecessary compilation each time in the loop:

{code-cell}
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()

"""

p2 = r"""
Just-in-time compilation
In this section, we will further explore how JAX works, and how we can make it performant. We will discuss the jax.jit() transformation, which will perform Just In Time (JIT) compilation of a JAX Python function so it can be executed efficiently in XLA.

How JAX transformations work
In the previous section, we discussed that JAX allows us to transform Python functions. JAX accomplishes this by reducing each function into a sequence of primitive operations, each representing one fundamental unit of computation.

One way to see the sequence of primitives behind a function is using jax.make_jaxpr():

import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
{ lambda ; a:f32[]. let
    b:f32[] = log a
    c:f32[] = log 2.0:f32[]
    d:f32[] = div b c
  in (d,) }
The JAX internals: The jaxpr language section of the documentation provides more information on the meaning of the above output.

Importantly, notice that the jaxpr does not capture the side-effect present in the function: there is nothing in it corresponding to global_list.append(x). This is a feature, not a bug: JAX transformations are designed to understand side-effect-free (a.k.a. functionally pure) code. If pure function and side-effect are unfamiliar terms, this is explained in a little more detail in 🔪 JAX - The Sharp Bits 🔪: Pure Functions.

Impure functions are dangerous because under JAX transformations they are likely not to behave as intended; they might fail silently, or produce surprising downstream errors like leaked Tracers. Moreover, JAX often can’t detect when side effects are present. (If you want debug printing, use jax.debug.print(). To express general side-effects at the cost of performance, see jax.experimental.io_callback(). To check for tracer leaks at the cost of performance, use with jax.check_tracer_leaks()).

When tracing, JAX wraps each argument by a tracer object. These tracers then record all JAX operations performed on them during the function call (which happens in regular Python). Then, JAX uses the tracer records to reconstruct the entire function. The output of that reconstruction is the jaxpr. Since the tracers do not record the Python side-effects, they do not appear in the jaxpr. However, the side-effects still happen during the trace itself.

Note: the Python print() function is not pure: the text output is a side-effect of the function. Therefore, any print() calls will only happen during tracing, and will not appear in the jaxpr:

def log2_with_print(x):
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))
printed x: Traced<~float32[]>with<DynamicJaxprTrace>
{ lambda ; a:f32[]. let
    b:f32[] = log a
    c:f32[] = log 2.0:f32[]
    d:f32[] = div b c
  in (d,) }
See how the printed x is a Traced object? That’s the JAX internals at work.

The fact that the Python code runs at least once is strictly an implementation detail, and so shouldn’t be relied upon. However, it’s useful to understand as you can use it when debugging to print out intermediate values of a computation.

A key thing to understand is that a jaxpr captures the function as executed on the parameters given to it. For example, if we have a Python conditional, the jaxpr will only know about the branch we take:

def log2_if_rank_2(x):
  if x.ndim == 2:
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
{ lambda ; a:i32[3]. let  in (a,) }
JIT compiling a function
As explained before, JAX enables operations to execute on CPU/GPU/TPU using the same code. Let’s look at an example of computing a Scaled Exponential Linear Unit (SELU), an operation commonly used in deep learning:

import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
3.72 ms ± 49.9 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
The code above is sending one operation at a time to the accelerator. This limits the ability of the XLA compiler to optimize our functions.

Naturally, what we want to do is give the XLA compiler as much code as possible, so it can fully optimize it. For this purpose, JAX provides the jax.jit() transformation, which will JIT compile a JAX-compatible function. The example below shows how to use JIT to speed up the previous function.

selu_jit = jax.jit(selu)

# Pre-compile the function before timing...
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
268 μs ± 1.86 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
Here’s what just happened:

We defined selu_jit as the compiled version of selu.

We called selu_jit once on x. This is where JAX does its tracing – it needs to have some inputs to wrap in tracers, after all. The jaxpr is then compiled using XLA into very efficient code optimized for your GPU or TPU. Finally, the compiled code is executed to satisfy the call. Subsequent calls to selu_jit will use the compiled code directly, skipping the python implementation entirely. (If we didn’t include the warm-up call separately, everything would still work, but then the compilation time would be included in the benchmark. It would still be faster, because we run many loops in the benchmark, but it wouldn’t be a fair comparison.)

We timed the execution speed of the compiled version. (Note the use of block_until_ready(), which is required due to JAX’s Asynchronous dispatch).

Why can’t we just JIT everything?
After going through the example above, you might be wondering whether we should simply apply jax.jit() to every function. To understand why this is not the case, and when we should/shouldn’t apply jit, let’s first check some cases where JIT doesn’t work.

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

jax.jit(f)(10)  # Raises an error
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function f at /tmp/ipykernel_1836/2956679937.py:3 for jit. This concrete value was not available in Python because it depends on the value of the argument x.
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

jax.jit(g)(10, 20)  # Raises an error
TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
The error occurred while tracing the function g at /tmp/ipykernel_1836/722961019.py:3 for jit. This concrete value was not available in Python because it depends on the value of the argument n.
See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError
The problem in both cases is that we tried to condition the trace-time flow of the program using runtime values. Traced values within JIT, like x and n here, can only affect control flow via their static attributes: such as shape or dtype, and not via their values. For more detail on the interaction between Python control flow and JAX, see Control flow and logical operators with JIT.

One way to deal with this problem is to rewrite the code to avoid conditionals on value. Another is to use special Control flow operators like jax.lax.cond(). However, sometimes that is not possible or practical. In that case, you can consider JIT-compiling only part of the function. For example, if the most computationally expensive part of the function is inside the loop, we can JIT-compile just that inner part (though make sure to check the next section on caching to avoid shooting yourself in the foot):

# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
Array(30, dtype=int32, weak_type=True)
Marking arguments as static
If we really need to JIT-compile a function that has a condition on the value of an input, we can tell JAX to help itself to a less abstract tracer for a particular input by specifying static_argnums or static_argnames. The cost of this is that the resulting jaxpr and compiled artifact depends on the particular value passed, and so JAX will have to re-compile the function for every new value of the specified static input. It is only a good strategy if the function is guaranteed to see a limited set of static values.

f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
10
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
30
To specify such arguments when using jit as a decorator, a common pattern is to use python’s functools.partial():

from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
30
JIT and caching
With the compilation overhead of the first JIT call, understanding how and when jax.jit() caches previous compilations is key to using it effectively.

Suppose we define f = jax.jit(g). When we first invoke f, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of f will reuse the cached code. This is how jax.jit makes up for the up-front cost of compilation.

If we specify static_argnums, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs. If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling jax.jit() on temporary functions defined inside loops or other Python scopes. For most cases, JAX will be able to use the compiled, cached function in subsequent calls to jax.jit(). However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined. This will cause unnecessary compilation each time in the loop:

from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
jit called in a loop with partials:
168 ms ± 115 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
jit called in a loop with lambdas:
167 ms ± 71.3 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
jit called in a loop with caching:
1.32 ms ± 2.64 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
"""

p3=r"""
Summarize the following text in one sentence:

The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.
In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that ``brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.
A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.
In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.
In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.
This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.
"""


# %%

s_p1 = sanitize_text_for_tokenizer(p1)
s_p2 = sanitize_text_for_tokenizer(raw_text) 
s_p3 = sanitize_text_for_tokenizer(p3)

