
#===

    Python Code:

    Can be also put in another file. or alternatively Below:

using PyCall # julia package to interact with python

py"""

latin_chars = 'abcdefghijklmnopqrstuvwxyz '

def normalise_filter_txt(txt):
    txt = txt.lower().strip()
    txt = ' '.join(txt.split())
    return ''.join([c for c in txt if c in latin_chars])

...

"""

===#


# void data for computations state
dataSTATE = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()
