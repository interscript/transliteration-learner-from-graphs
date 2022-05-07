

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Agent.jl")

include("src/Rules.jl")


using PyCall

hazm = pyimport("hazm")

PATH_HAZM = "resources/postagger.model"
PATH_FLEXICON = "resources/"

stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM);




function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the train model"
        "--text"
            help = "farsi text to be transliterated"

        # additional fields can be added
        # "--pos-tagging"
        #     help = "PoS tagging, as found by hazm POSTagger"

    end

    parse_args(s)

end


# parse commands
parsedArgs = parse_commandline()


# load brain data
data = deserialize(parsedArgs["path-model"])

entryBrain = data[:entry]
dicBRAINS = data[:dicBrains]
df_Nodes = data[:df_Nodes]
graph = dicBRAINS[entryBrain]


data = Dict{String, Any}(
            "txt" => parsedArgs["text"],
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain)
            # "pos" => parsedArgs["pos-tagging"]) # current brain or graph
            # additional states can be added

data |>
    println


# run agent
runAgent(graph, dicBRAINS, df_Nodes, data) |>
    println


"####################################################################" |> println
"###########################  END ###################################" |> println
"####################################################################" |> println
