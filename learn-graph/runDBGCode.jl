
using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Agent.jl")
include("src/Rules.jl")



function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the train model"
        "--text"
            help = "farsi text to be transliterated"

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
            "text" => parsedArgs["text"],
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain)


# run agent
runAgent(graph, dicBRAINS, df_Nodes, data) |>
    println


"####################################################################" |> println
"###########################  END ###################################" |> println
"####################################################################" |> println
