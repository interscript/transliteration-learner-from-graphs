
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
            d["res"] = d["txt"]
            d
        end, # identity
        Dict(:in => ["txt"], :out => ["res"]))


dicCODE["bind transliterated words together"] =
    #===    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["txt"] = join(d["l_transl_wrds"], " ");
         d),
        Dict(:in => ["l_transl_wrds"], :out => ["txt"]))

        dicCODE["apply mappings on each word"] =
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
                Dict(:in => ["l_wrds"],
                     :out => ["l_transl_wrds"]))
