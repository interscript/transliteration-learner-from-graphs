
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


# computations state
dataSTATE = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()


# Node terminating computation
dicCODE["done, terminate"] =
    #===
        Basic form of functors:
            d: data
            e: dicBRAINS
            f: df_Nodes

        Inputs and Outputs are specified
        :in => "l_transliterated" # list of
        :out => "res" # field expected at end of (sub)sequence
    ===#
    Functor((d,e=nothing,f=nothing) ->
        begin
            d["res"] = d["transl_txt"]
            # clean up if necessary
            delete!(d, "transl_txt")
            # or
            [delete!(d, f) for f in ["l_transl_wrds", "transl_txt", "l_wrds"]]
            # return dictionary
            d
        end, # identity
        Dict(:in => ["transl_txt"], :out => ["res"]))


dicCODE["bind transliterated words together"] =
    #===    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["transl_txt"] = join(d["l_transl_wrds"], " ");
         d),
        Dict(:in => ["l_transl_wrds"], :out => ["transl_txt"]))

dicCODE["process each word with mapping"] =
    Functor((d,e=nothing,f=nothing) ->
        begin
            d["l_transl_wrds"] =
            map(wrd ->
                begin
                    dd = copy(dataSTATE)
                    dd["wrd"] = wrd
                    interfaceName = "mapping"
                    node = e[interfaceName]
                    runAgent(node, e, f, dd)
                end,
                d["l_wrds"])
                d
        end,
            Dict(:in => ["l_wrds"],:out => ["l_transl_wrds"]))

dicCODE["has word the char z?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = occursin("z", d["wrd"]) ?
                    "yes" : "no";
         d),
         Dict(:in => ["wrd"], :out => ["wrd", "state"]))


dicCODE["map z to @"] =
    Functor((d,e=nothing,f=nothing) ->
        (wrd = d["wrd"];
         d["wrd"] =
            read(pipeline(`echo $wrd`, `sed s/z/@/g`), String);
         d),
         Dict(:in => ["wrd"], :out => ["wrd"]))


using PyCall

py"""
d_maps = {'z': 'a',
          'y': 'z',
          'x': 'y',
          'w': 'x',
          'v': 'w',
          'u': 'v',
          't': 'u',
          's': 't',
          'r': 's',
          'q': 'r',
          'p': 'q',
          'o': 'p',
          'n': 'o',
          'm': 'n',
          'l': 'm',
          'k': 'l',
          'j': 'k',
          'i': 'j',
          'h': 'i',
          'g': 'h',
          'f': 'g',
          'e': 'f',
          'd': 'e',
          'c': 'd',
          'b': 'c',
          'a': 'b',
          '@': 'a'}

"""

dicCODE["map all letters utilising table and @ to a"] =
    #===
        We use a python dictionary to generate the mappings.
        Alternative snippet structure with begin; end;
    ===#
    Functor((d,e=nothing,f=nothing) ->
        begin
            wrd = d["wrd"]
            d["res"] = join([py"""d_maps"""[string(c)] for c in wrd
                     if c!='\n'])
                         d
        end,
            Dict(:in => ["wrd"], :out => ["res"]))

py"""
latin_chars = 'abcdefghijklmnopqrstuvwxyz '

def normalise_filter_txt(txt):
    txt = txt.lower().strip()
    txt = ' '.join(txt.split())
    return ''.join([c for c in txt if c in latin_chars])

"""

dicCODE["normalize the text!"] =
    #===
        Example calling a python function
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["txt"] = py"""normalise_filter_txt"""(d["txt"]);
         d),
            Dict(:in => ["txt"], :out => ["txt"]))


dicCODE["tokenize the text!"] =
#===
    Example running julia code
        Function
            :in => txt
            :out => l_wrds
===#
Functor((d,e=nothing,f=nothing) ->
    (d["l_wrds"] = split(d["txt"]);
     d),
        Dict(:in => ["txt"], :out => ["l_wrds"]))
