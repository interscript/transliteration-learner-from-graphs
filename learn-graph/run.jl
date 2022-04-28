

using Graphs
using CSV
using DataFrames
using ArgParse
using Serialization


include("src/Graphs.jl")
include("src/Rules.jl")
include("src/Agent.jl")
include("src/Metrics.jl")

using Logging
Logging.disable_logging(Logging.Info)

using PyCall

hazm = pyimport("hazm")

PATH_HAZM = "resources/postagger.model"
PATH_FLEXICON = "resources/"


stemmer = hazm.Stemmer()
lemmatizer = hazm.Lemmatizer()
normalizer = hazm.Normalizer()
tagger = hazm.POSTagger(model=PATH_HAZM)


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
dataM = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "pre_pos" => nothing,
            "state" => nothing, # used for messages back to system
            "brain" => entryBrain) # current brain or graph


VOCABFARSI = " !\"()+-./0123456789:<>ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}«»،؛؟ءآأؤئابةتثجحخدذرزسشصضطظعغـفقلمنهوَِْپچژکگی\u200c"
VOCABFARSI =" ءآأؤئابةتثجحخدذرزسشصضطظعغـفقلمنهوَِْپچژکگی \u200c "

function processPOS(pos)

    l_supported_POS = vcat(py"""l_PoS""",
                       collect(keys(py"""d_map_FLEXI""")),
                       collect(keys(py"""d_map_HAZM""")))

    if !(pos in l_supported_POS)

        @error pos, "pos unrecognised, needs to be within: ", l_supported_POS
        exit()

    else

        if pos in collect(keys(py"""d_map_FLEXI"""))

            pos = py"""d_map_FLEXI"""[pos]

        elseif pos in collect(keys(py"""d_map_HAZM"""))

            pos = py"""d_map_HAZM"""[pos]

        end

    end

    pos

end


m = 1
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
                                                join(filter(c -> c in VOCABFARSI, d[1]), "") : d[1];
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
    
    ProgressBar(readlines(parsedArgs["file-name"], keep=true) |> 
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
                                                join(filter(c -> c in VOCABFARSI, d[1]), "") : d[1];
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
