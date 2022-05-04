

# Include python library
using PyCall

# load python library
include("libs/code_py.jl")


# void data for computations state
dataM = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()


#####################
# mappings          #
#####################


dicCODE["done, do nothing!"] =
    #===
        Basic form of functors:
            d: data
            e: dicBRAINS
            f: df_Nodes
    ===#
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))


dicCODE["map each char with mapping"] =
    #===

    ===#
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))



#####################
# preprocessor      #
#####################


dicCODE["normalize the text!"] =
    #===
        Example calling a python function
    ===#
    Functor((d,e=nothing,f=nothing) ->
        (d["text"] = py"""normalise_filter_text"""(d["text"]);
         d),
        Dict(:in => ["text"], :out => ["text"]))


#####################
# transliteration   #
#####################

dicCODE["normalize the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))
