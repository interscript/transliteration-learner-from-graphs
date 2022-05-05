
#===

    Python Code:

    Can be also put in another file.

===#

using PyCall # julia package to interact with python

py"""

latin_chars = 'abcdefghijklmnopqrstuvwxyz '

def normalise_filter_txt(txt):
    txt = txt.lower().strip()
    txt = ' '.join(txt.split())
    return ''.join([c for c in txt if c in latin_chars])

d_maps = {'z': 'a',
          'y': 'x',
          'x': 'w',
          'w': 'v',
          'v': 'u',
          'u': 't',
          't': 's',
          's': 'r',
          'r': 'q',
          'q': 'p',
          'p': 'o',
          'o': 'n',
          'n': 'm',
          'm': 'l',
          'l': 'k',
          'k': 'j',
          'j': 'i',
          'i': 'h',
          'h': 'g',
          'g': 'f',
          'f': 'e',
          'e': 'd',
          'd': 'e',
          'c': 'd',
          'b': 'c',
          'a': 'b'}

"""

# void data for computations state
dataM = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()
