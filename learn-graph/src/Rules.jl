

#include("hazm/py_code.jl")


dataM = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "state" => nothing,
            "brain" => nothing);



dicCODE = Dict{String, Functor}()


#####################
# preprocessor      #
#####################

dicCODE["normalize the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))


#####################
# mappings          #
#####################

dicCODE["done, do nothing!"] =
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))

dicCODE["map each char with mapping"] =
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))


#####################
# transliteration   #
#####################

dicCODE["normalize the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        d,
        Dict(:in => ["text"], :out => ["text"]))
