

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization
using Logging
Logging.disable_logging(Logging.Info)

#using PyCall


include("src/Graphs.jl")
include("src/Agent.jl")
include("src/Metrics.jl")
include("src/Rules.jl")


function parse_commandline()

    s = ArgParseSettings()

    @add_arg_table! s begin

        "--path-model"
            help = "path to the train model"

        "--file-name"
            help = "file-name to be transliterated \n
                    excepted if file-name = data/test.csv \n
                    in that case, tests are performed."

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


# prepare data
dataM = dataSTATE


dicParams = YAML.load_file("../config/params.yml")
SOURCECHARS = dicParams["transliteration"]["SourceCHARS"]


if parsedArgs["file-name"] in ["data/test.csv", "test"] # Run the test


    df_Test = DataFrame(CSV.File("data/test.csv"))

    df_Test[!,"transModel"] =
        map(d -> d |>
            py"""normalise""" |>
                hazm.word_tokenize |>
                    tagger.tag |>
                        (D -> map(d -> (dd = copy(dataM);
                                        dd["pos"] = processPOS(d[2]);
                                        dd["word"] = d[2] != "Punctuation" ?
                                                join(filter(c -> c in SOURCECHARS, d[1]), "") : d[1];
                                        dd["state"] = nothing;

                            try

                                # println(dd["word"], " : ", dd["pos"])
                                dd["pos"] == "Punctuation" ?
                                    dd["word"] : runAgent(graph, dicBRAINS, df_Nodes, dd) |>
                                            (w -> replace(w, "-''"=>"", "-'"=>""));
                        

                            catch

                                println("DBG:: ", dd["word"], " : ", dd["pos"]);
                                dd["word"]

                            end), D)) |>
                                (L -> join(L, " ")), 
            df_Test[!,"orig"])

    ids = evaluation(df_Test[!, "trans"], df_Test[!, "transModel"], df_Test[!, "orig"])

    df_Bugs = df_Test[ids,:]

    println("error summary in: data/test_debug.csv")
    CSV.write("data/test_debug.csv", df_Bugs)


else # transliterate the file

    using ProgressBars
    
    function preprocessData(data, window=6, space=4)
    
        d_data = []
        for d in ProgressBar(data)
    
            w = split(d)
            for i=1:space:length(w) 

                push!(d_data, 
                      w[i:min(end,i+window)])
        
                if i+window > length(w)
                    break
                end
        
            end
        end
        d_data
        
    end
    
    ProgressBar(readlines(parsedArgs["file-name"], keep=true) |> 
                            preprocessData |>
                                (D -> filter(s -> strip(s) != "", D))) |> 
      (D ->
        map(d ->
            (println("f::"*chomp(d));
             chomp(d) |>
                py"""normalise""" |>
                    hazm.word_tokenize |>
                        tagger.tag |>
                            (Ws -> map(d -> (dd = copy(dataM);
                                        dd["pos"] = processPOS(d[2]);
                                        dd["word"] = d[2] != "Punctuation" ?
                                                join(filter(c -> c in SOURCECHARS, d[1]), "") : d[1];
                                        dd["state"] = nothing;

                                        try
                                            
                                            dd["pos"] == "Punctuation" || strip(dd["word"]) == "" ?
                                                dd["word"] : runAgent(graph, dicBRAINS, df_Nodes, dd)

                                        catch

                                            println("DBG:: ", dd["word"], " : ", dd["pos"]);
                                            dd["word"];
                                            ### exit(); # ""

                                        end),
                                        # runAgent(graph, dicBRAINS, df_Nodes, dd)
                                    Ws)) |>
                            (L -> "t::"*join(L, " ")) |>
                                println),
            D))

end
