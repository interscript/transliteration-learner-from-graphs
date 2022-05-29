

include("hazm/py_code.jl")


dataM = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "state" => nothing,
            "brain" => nothing);

dataN = Dict{String, Any}(
            "word" => nothing,
            "pos" => nothing,
            "pre_pos" => nothing,
            "state" => nothing,
            "brain" => "transliterator");


dicCODE = Dict{String, Functor}()


function processPOS(pos)

    l_supported_POS = vcat(py"""l_PoS""",
                       collect(keys(py"""d_map_FLEXI""")),
                       collect(keys(py"""d_map_HAZM""")))

    if !(pos in l_supported_POS)

        @error "pos unrecognised, needs to be within: ", l_supported_POS
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


#####################
# preprocessor      #
#####################

dicCODE["normalize the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["text"] = py"""normalise"""(d["text"]); d),
        Dict(:in => ["text"], :out => ["text"]))

dicCODE["tokenize the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["text"] = hazm.word_tokenize(d["text"]); d),
        Dict(:in => ["text"], :out => ["text"]))

dicCODE["recognize parts of speech in the text!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["wrd+pos"] = tagger.tag(d["text"]); d),
        Dict(:in => ["text"], :out => ["wrd+pos"]))

dicCODE["run transliterator on each word!"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         l_res = map(wp -> (wrd = wp[1];
                            pos = wp[2] |> processPOS;
                            dd = copy(dataM);
                            dd["word"] = wrd;
                            dd["pos"] = pos;
                            interfaceName = "transliterator";
                            dd["brain"] = interfaceName;
                            node = copy(e[interfaceName]);
                            dd["pos"] == "Punctuation" ?
                                dd["word"] : runAgent(node, e, f, dd) |>
                                    (w -> replace(w, "-''"=>"", "-'"=>""))),
                    d["wrd+pos"]);
         d["res"] = join(l_res, " ");
         d["brain"] = bN;
         d),
            Dict(:in => ["wrd+pos"], :out => ["res"]))


#####################
# transliterator    #
#####################

dicCODE["do nothing!"] =
    Functor((d,e=nothing,f=nothing) -> d,
        Dict(:in => ["word"], :out => ["word"]))

dicCODE["change all instances of ي and ك and ۀ in the text to ی and ک and هٔ"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["word"]=py"""normalise"""(d["word"]); d),
            Dict(:in => ["word"], :out => ["word"]))

dicCODE["is the word found in the db?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["data"]=py"""search_db"""(d["word"], d["pos"]);
         d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["word", "pos"], :out => ["data", "state"]))

dicCODE["is it a verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = d["pos"] == "Verb" ? "yes" : "no"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["lemmatize it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = d["word"] |> lemmatizer.lemmatize |>
            (S -> split(S, "#")) |>
                (S -> filter(c -> c != "", S)) |>
                    (S -> join(S, "#")); d),
            Dict(:in => ["word"], :out => ["lemma"]))

dicCODE["includes underscores?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = contains(d["word"], "_") ? "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["does only one of the verb roots exist in the verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 1 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))

dicCODE["output it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (haskey(d, "res") ?
            "" :
            (typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                  d["res"] = py"""return_highest_search_pos"""(d["data"], d["pos"]) :
                  d["res"] = d["word"]); d),
            Dict(:in => ["data"], :out => []))

dicCODE["collision?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = length(d["data"]) == 1 ? "no" : "yes"; d),
            Dict(:in => ["data"], :out => ["state"]))

dicCODE["does it include u200c?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = contains(d["word"], "\u200c") ? "yes" : "no"; d),
        Dict(:in => ["word"], :out => ["state"]))

#===
    new changes...
===#

dicCODE["is the segment before it نمی?"] =
    Functor((d,e=nothing,f=nothing) ->
        (lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         d["state"] = join(lStr[1:idx-1], "") == "نمی" ? "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["is the segment before it می?"] =
    Functor((d,e=nothing,f=nothing) ->
        (lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         d["state"] = join(lStr[1:idx-1], "") == "می" ? "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["does the segment after it start with ها?"] =
    Functor((d,e=nothing,f=nothing) ->
        (lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         d["state"] = join(lStr[idx+1:min(end,idx+2)], "") == "ها" ?
                "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["transliterate the segment before u200c and mark the segment after u200c as suffix."] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         # pre u200c
         dd = copy(dataN);
         dd["word"] = join(lStr[1:idx-1], "");
         dd["pos"] = d["pos"];
         interfaceName = "transliterator";
         node = e[interfaceName];
         pre_res = runAgent(node, e, f, dd);
         d["res_root"] = pre_res;
         # post u200c
         d["word"] = join(lStr[idx+1:end], "");
         d["affix"] = d["word"];
         d["suffix"] = d["word"];
         d["brain"] = bN; d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["transliterate each side of it separately in proper order"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         dd = copy(dataN);
         # pre u200c
         dd["word"] = join(lStr[1:idx-1], "");
         dd["pos"] = d["pos"];
         interfaceName = "transliterator";
         node = copy(e[interfaceName]);
         d["res"] = runAgent(node, e, f, dd);
         # post u200c
         dd = copy(dataN);
         dd["word"] = join(lStr[idx+1:end], "");
         dd["pos"] = d["pos"];
         dd["hassemispace"] = true;
         node = copy(e[interfaceName]);
         w = runAgent(node, e, f, dd);
         d["res"] = d["res"] * w;
         d["brain"] = bN; d),
            Dict(:in => ["word"], :out => ["state"]))

# dicCODE["transliterate the segment after u200c as a verb and add mi to the beginning of it"] =
dicCODE["transliterate the segment after u200c as a verb, starting at \"lemmatize it!\" and add mi to the beginning of it"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         prev_wrd = join(lStr[1:idx-1], "");
         wrd = join(lStr[idx+1:end], "");
         dd = copy(dataN);
         dd["word"] = wrd;
         dd["pos"] = "Verb";
         interfaceName = "transliterator"; # "verb-handler"
         node = e[interfaceName];
         res = wrd != "" ? "mi"*runAgent(node, e, f, dd) : "mi";
         d["res"] = res;
         d["brain"] = bN;d),
            Dict(:in => ["word"], :out => ["res"]))

 dicCODE["transliterate the segment after u200c as a verb, starting at \"lemmatize it!\" and add nemi to the beginning of it"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         lStr = collect(d["word"]);
         idx = indexin('\u200c', lStr)[1];
         wrd = join(lStr[idx+1:end], "");
         dd=copy(dataN);
         dd["word"] = wrd;
         dd["pos"] = "Verb";
         interfaceName = "transliterator"; #"verb-handler"
         node = e[interfaceName];
         d["res"] = "nemi"*runAgent(node, e, f, dd);
         d["brain"] = bN;d),
            Dict(:in => ["word"], :out => ["state"]))

dicCODE["output its transliteration!"] =
    Functor((d,e=nothing,f=nothing) ->
        (if !haskey(d, "res")
            typeof(d["data"]) == Vector{Dict{Any, Any}} ?
                (v = py"""return_highest_search_pos"""(d["data"], d["pos"]);
                 d["res"] = v[1]; d["SynCatCode"] = v[2]) :
                 d["res"] = d["word"];
        end;
        d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return its transliteration!"] = dicCODE["output its transliteration!"]


dicCODE["stem it!"] =
    Functor((d,e=nothing,f=nothing) -> (d["lemma"] = stemmer.stem(d["word"]); d),
            Dict(:in => ["word"], :out => ["lemma"])) # lemma

dicCODE["is the verb root found in the db?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["data"]=py"""search_db"""(d["lemma"], d["pos"]);
         d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["lemma", "pos"], :out => ["data", "state"]))

dicCODE["does the root of the word exist in the database?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["data"]=py"""search_db"""(d["lemma"], d["pos"]);
         d["state"] = typeof(d["data"]) != String ? "yes" : "no"; d),
            Dict(:in => ["lemma", "pos"], :out => ["data", "state"]))

dicCODE["transliterate each side of underscore separately in proper order"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         d["res"] = map(w -> (dd = copy(dataN);
                              dd["word"] = w;
                              dd["pos"] = d["pos"];
                              interfaceName = "transliterator";
                              node = e[interfaceName];
                              runAgent(node, e, f, dd)),
                    split(d["word"], "_")) |>
                        (D -> join(D, ""));
         d["root"] = d["word"]; # to end computation
         d["brain"] = bN;
         d),
            Dict(:in => ["lemma"], :out => ["res"]))


# collision-handler

dicCODE["is there an instance of the word with the desired pos?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = typeof(d["data"]) != String ?
            py"""has_entries_search_pos"""(d["data"], d["pos"]) : "no"; d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["is there only one instance of the word with the desired pos?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = py"""has_only_one_search_pos"""(d["data"], d["pos"]); d),
            Dict(:in => ["data", "pos"], :out => ["state"]))

dicCODE["return the transliteration of the instance with the desired pos!"] =
    Functor((d,e=nothing,f=nothing) ->
        ((d["res"], d["SynCatCode"]) = py"""return_highest_search_pos"""(d["data"], d["pos"]);
         d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the desired pos that has the highest frequency!"] =
    Functor((d,e=nothing,f=nothing) ->
        (v = py"""return_highest_search_pos"""(d["data"], d["pos"]);
         d["res"] = v[1]; d["SynCatCode"] = v[2]; d),
            Dict(:in => ["data", "pos"], :out => ["res"]))

dicCODE["return the transliteration of the instance with the highest frequency!"] =
    Functor((d,e=nothing,f=nothing) -> (
        v = py"""return_highest_search_pos"""(d["data"], d["pos"]);
        if typeof(v) == String
            d["res"] = v
        else
            d["res"] = v[1]
            d["SynCatCode"] = v[2]
        end; d),
            Dict(:in => ["data"], :out => ["res"]))


# Full Model

dicCODE["return \"id\""] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "id"; d),
            Dict(:in => [], :out => []))

dicCODE["terminator"] =
    Functor((d,e=nothing,f=nothing) -> d,
            Dict(:in => [], :out => []))


# affix-handler

dicCODE["is it ست?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ست" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ی?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ی" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ات?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ات" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ان?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ان" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ش?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ش" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it م?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "م" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))


dicCODE["is it مان?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "مان" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it می?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "می" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ون?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ون" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ید?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ید" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it یم?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "یم" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it ن?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] == "ن" ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["is it بی or نی?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["affix"] in ["بی", "نی"] ? "yes" : "no"; d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["return \"st\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "st"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"ast\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "ast"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"i\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "i"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"ye\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "ye"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"at\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "At"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"'at\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "'at"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yan\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yAn"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"an\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "An"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"as\""] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "aS"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"es\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "eS"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"om\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "om"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"am\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "am"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"man\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "mAn"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"na\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "na"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"eman\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "emAn"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"omi\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "omi"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"mi\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "mi"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yun\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yun"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"un\""] =
        Functor((d,e=nothing,f=nothing) -> (d["res"] = "un"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yad\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yad"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"im\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "im"; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["return \"yam\""] =
    Functor((d,e=nothing,f=nothing) -> (d["res"] = "yam"; d),
            Dict(:in => [], :out => ["res"]))


dicCODE["is it a suffix?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if haskey(d,"suffix")
                d["suffix"] == d["affix"] ? "yes" : "no"
            elseif haskey(d, "hassemispace")
                d["hassemispace"] ? "yes" : "no"
            else
                "no"
            end; d),
            Dict(:in => ["word", "affix"], :out => ["state"]))

dicCODE["is there only one instance of the affix?"] =
    Functor((d,e=nothing,f=nothing) ->
    (if !haskey(d, "data")
        d["data"] = py"""affix_search"""(d["affix"])
     end;
     d["state"] = py"""has_only_one_search_pos"""(d["data"]); d),
            Dict(:in => ["affix"], :out => ["state"]))

dicCODE["use it! "] =
    Functor((d,e=nothing,f=nothing) ->
            (d["lemma"] = filter(x -> contains(d["word"], x),
                                 split(d["lemma"], "#"))[1];
             d),
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["use it!"] = dicCODE["use it! "]

dicCODE["is the word before it a verb?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if d["affix"] == get(d, "suffix", false) # affix?
                haskey(d,"lemma") && d["pos"] == "Verb" ? "yes" : "no";
            elseif d["affix"] == get(d, "prefix", "nothing") # suffix
                if haskey(d, "pre_pos")
                    d["pre_pos"] == "Verb" ? "yes" : "no"
                else
                    d["pos"] == "Verb" ?  "yes" : "no"
                end;
            else
                "no"
            end; d),
            Dict(:in => [], :out => ["state"]))

dicCODE["is the word to-which it's attached, a noun?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["pos"] == "Noun" ? "yes" : "no"; d),
            Dict(:in => ["pos"], :out => ["state"]))

dicCODE["is the word to-which it's attached, a number or چند?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = contains(d["word"], "چند") ? "yes" :
                d["pos"] == "Number" ? "yes" : "no";
         d),
            Dict(:in => ["pos", "affix"], :out => ["state"]))

dicCODE["is the verb root to-which it's attached, marked as v2 in the database?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["SynCatCode"]=="V2" ? "yes" : "no"; d),
            Dict(:in => ["SynCatCode"], :out => ["state"]))

dicCODE["does the verb root to-which it's attached, end in any of the /e, a, u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["res_root"][end] in ['e', 'A', 'u'] ? "yes" : "no"; d),
            Dict(:in => ["res_root"], :out => ["state"]))

dicCODE["is there a space or semi-space before it?"] =
    Functor((d,e=nothing,f=nothing) ->
        (n = length(d["affix"]);
         if haskey(d, "hassemispace")
             d["state"] = "yes"
             @goto hassemispace
         end;
         idx = first(findlast(d["affix"], d["word"]));
         if idx > 1
             d["word"][idx-n:idx] == " " ?
                d["state"] = "yes" :
                    if n > 4
                        d["state"] =
                            d["word"][n-3:n-1+length(d["affix"])] == string('\u200c', d["affix"]) ?
                                "yes" : "no"
                    else
                            d["state"] = "no"
                    end
          else
             d["state"] = "no"
          end;
          @label hassemispace; # jair
          d),
            Dict(:in => ["word", "affix"], :out => ["state"]))


dicCODE["return its transliteration in affixes"] =
    Functor((d,e=nothing,f=nothing) ->
        (wrd = haskey(d, "affix") ? d["affix"] : d["word"];
         if haskey(d, "data")
             if typeof(d["data"]) != Vector{Dict{Any, Any}}
                 d["data"] = py"""affix_search"""(wrd)
             end
         else
              d["data"] = py"""affix_search"""(wrd)
         end;
         d["res"] = d["data"][1]["PhonologicalForm"];
         d),
            Dict(:in => ["data"], :out => ["res"]))


dicCODE["is the prefix ب or بی?"] =
    Functor((d,e=nothing,f=nothing) ->
                (wrd = d["word"];
                 root = filter(x -> contains(wrd, x), split(d["lemma"], "#"))[1];
                 n = length(collect(root));
                 nWord = length(collect(wrd));
                 idx = nothing;
                 for i=1:nWord-n+1
                     if join(collect(d["word"])[i:i+n-1], "") == root
                         idx = i
                         break
                     end
                 end;
                 d["state"] = join(collect(d["word"])[1:idx-1], "") in  ["ب", "بی"] ?
                            "yes" : "no"; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["do both verb roots exist in the verb?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(filter(x -> occursin(x, d["word"]),
                                      split(d["lemma"], "#"))) == 2 ? "yes" : "no"; d),
            Dict(:in => ["word", "lemma"], :out => ["state"]))


dicCODE["use the second verb root!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = split(d["lemma"], "#")[2]; d),
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["use the first verb root!"] =
    Functor((d,e=nothing,f=nothing) ->
            (d["lemma"] = split(d["lemma"], "#")[1]; d),
            Dict(:in => ["lemma"], :out => ["res"]))


dicCODE["is there an آ in the verb roots?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = contains(d["lemma"], "آ") ?  "yes" : "no"; d),
            Dict(:in => ["lemma"], :out => ["state"]))


dicCODE["change the first آ in the verb root(s) to ا."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = split(d["lemma"], "#") |>
            (Ls -> map(x -> (idx = findfirst("آ", x) |> first;
                             string(replace(x[1:idx], "آ" => "ا"), x[idx+2:end])),
                       Ls) |>
                    (L -> join(L, "#")));
             d),
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["undo the change to the first verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = replace(split(d["lemma"], "#")[1], "ا" =>
                            "آ"); d),
            Dict(:in => ["lemma"], :out => ["lemma"]))

dicCODE["undo the change to the second verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["lemma"] = replace(split(d["lemma"], "#")[2], "ا" =>
            "آ"); d),
            Dict(:in => ["lemma"], :out => ["root"]))


dicCODE["does the transliteration of the segment before it end in any of the /a, i, u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if !isnothing(get(d, "segm", nothing))
                d["segm"][end] in ['A', 'u', 'i'] ? "yes" : "no";
            elseif haskey(d, "l_res")
                d["l_res"][end] in ['A', 'u', 'i'] ? "yes" : "no";
            elseif (haskey(d, "res_root") && haskey(d, "suffix") && length(d["res_root"]) > 0)
                d["res_root"][end] in ['A', 'u', 'i'] ? "yes" : "no";
            else
                "no"
            end; d),
            Dict(:in => [], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in any of the /a,e,o,a,u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if !isnothing(get(d, "segm", nothing))
                d["segm"][end] in ['A', 'a', 'e', 'o', 'u'] ? "yes" : "no";
            elseif haskey(d, "l_res")
                d["l_res"][end] in ['A', 'a', 'e', 'o', 'u'] ? "yes" : "no";
            elseif (haskey(d, "res_root") && haskey(d, "suffix") && length(d["res_root"]) > 0)
                d["res_root"][end] in ['A', 'a', 'e', 'o', 'u'] ? "yes" : "no";
            else
                "no"
            end; d),
            Dict(:in => [], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in /i/?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = if !isnothing(get(d, "segm", nothing))
             d["segm"][end] == 'i' ? "yes" : "no"
         elseif haskey(d, "l_res")
             d["l_res"][end][end] == 'i' ? "yes" : "no"
         elseif haskey(d, "res_root")
             if d["res_root"] == ""
                 "no"
             else
                 d["res_root"][end] == 'i' ? "yes" : "no"
             end
         else
            "no"
         end; d),
            Dict(:in => [], :out => ["state"]))


dicCODE["does the transliteration of the segment before it end in any of the /a,e,o,a,i,u/ sounds?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =  if !isnothing(get(d, "segm", nothing))
            d["segm"][end] in ['A', 'a', 'e', 'o','u', 'i'] ? "yes" : "no"
         elseif haskey(d, "l_res")
            d["l_res"][end][end] in ['A', 'a', 'e', 'o', 'u', 'i'] ? "yes" : "no"
         elseif haskey(d, "res_root")
            d["res_root"][end] in ['A', 'a', 'e', 'o', 'u'] ? "yes" : "no"
         else
             "no"
         end;
         d),
            Dict(:in => [], :out => ["state"]))


dicCODE["is there anything after the word root?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = if length(d["lemma"]) == length(d["word"])
            "no"
            else
                begin
                    lemma = d["lemma"]
                    lemma2 = replace(d["lemma"], "آ"  =>
                             "ا")
                    if contains(d["word"], lemma2)
                        last(findfirst(lemma2, d["word"])) < last(findlast(d["word"], d["word"])) ? "yes" : "no"
                    elseif contains(d["word"], lemma)
                        last(findfirst(lemma, d["word"])) < last(findlast(d["word"], d["word"])) ? "yes" : "no"
                    else
                        "no"
                    end
                end
            end; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["is there anything before the word root?"] =
    Functor((d,e=nothing,f=nothing) ->
        (lemma = replace(d["lemma"], "آ"  =>
                        "ا");
         d["state"] = if contains(d["word"], lemma)
            d["state"] = 1 == first(findfirst(lemma, d["word"])) ? "no" : "yes"
         else
            "no"
         end; d),
            Dict(:in => ["lemma", "word"], :out => ["state"]))


dicCODE["is it a single letter?"] =
    Functor((d,e=nothing,f=nothing) -> (d["state"] = length(d["word"]) == 1 ? "yes" : "no"; d),
            Dict(:in => ["word"], :out => ["state"]))


dicCODE["is it found in affixes?"] =
    Functor((d,e=nothing,f=nothing) ->
        (data = py"""affix_search"""(
            haskey(d, "affix") ? d["affix"] : d["word"]);
         d["state"] = if typeof(data) == String
                "no"
            else
                length(data) > 0 ? "yes" : "no"
         end;
         d),
        Dict(:in => [], :out => ["state"]))


dicCODE["return the transliteration with t as its pos"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = py"""get_in_db"""(haskey(d, "affix") ?
                d["affix"] : d["word"], "T")[1];
                d),
            Dict(:in => ["word"], :out => ["res"]))


dicCODE["return its transliteration then omit the ' symbol in the beginning of the word root that comes after it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = string(collect(d["data"])[1]["PhonologicalForm"], "-'"); d),
            Dict(:in => ["data"], :out => ["res"]))


dicCODE["is the word root, رو recognized as a verb?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["pos"] == "Verb" && d["lemma"] == "رو" ? "yes" : "no"; d),
            Dict(:in => ["data", "pos"], :out => ["state", "res"]))

dicCODE["is the word root, شو recognized as a verb?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["pos"] == "Verb" && d["lemma"] == "شو" ? "yes" : "no"; d),
            Dict(:in => ["data", "pos"], :out => ["state", "res"]))

dicCODE["is the word root, شنو recognized as a verb?"] =
Functor((d,e=nothing,f=nothing) ->
    (d["state"] = d["pos"] == "Verb" && d["lemma"] == "شنو" ? "yes" : "no"; d),
        Dict(:in => ["data", "pos"], :out => ["state", "res"]))

dicCODE["change the word root's transliteration from /rav/ to /ro/"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = replace(d["res"], "rav" => "ro"); d),
            Dict(:in => ["res"], :out => ["res"]))

dicCODE["change the word root's transliteration from /senav/ to /seno/"] =
Functor((d,e=nothing,f=nothing) ->
    (d["res"] = replace(d["res"], "Senav" => "Seno"); d),
        Dict(:in => ["res"], :out => ["res"]))

dicCODE["change its transliteration from /rav/ to /ro/"] =
    dicCODE["change the word root's transliteration from /rav/ to /ro/"]

dicCODE["change the word root's transliteration from /sav/ to /so/"] =
Functor((d,e=nothing,f=nothing) ->
    (d["res"] = replace(d["res"], "sav" => "so"); d),
        Dict(:in => ["res"], :out => ["res"]))

dicCODE["mark it as prefix"] =
    Functor((d,e=nothing,f=nothing) ->
        (nWord = length(collect(d["word"]));
         # trying to solve all the cases
         if haskey(d, "d_substring") #length(lemmas) == 0
             lemma = d["d_substring"]["root"]
         else
             lemmas = split(d["lemma"], "#") |>
                      (L -> map(l -> [l, replace(l, "آ" =>
                                     "ا")],
                                L)) |>
                          (L -> vcat(L...)) |>
                              (L -> filter(l -> contains(d["word"], l), L))
             lemma = lemmas[1]
         end;

         n = length(collect(lemma));
         idx = nothing;
         for i=reverse(1:nWord-n+1)
             if join(collect(d["word"])[i:i+n-1], "") == lemma
                 idx = i
                 break
             end
         end;

         d["suffix"] = join(collect(d["word"])[idx+n:end], "");
         d["prefix"] = join(collect(d["word"])[1:idx-1]);
         d["affix"] = d["prefix"];
         d["data"] = py"""affix_search"""(d["prefix"]);
         d["res_root"] = haskey(d, "res_root") ? d["res_root"] : d["res"];
         delete!(d, "res");
         d),
            Dict(:in => ["word"], :out => ["prefix"]))

dicCODE["mark it as suffix"] =
    Functor((d,e=nothing,f=nothing) ->
      (nWord = length(collect(d["word"]));
       # trying to solve all the cases
       if haskey(d, "d_substring") #length(lemmas) == 0
           lemma = d["d_substring"]["root"]
       else
           lemmas = split(d["lemma"], "#") |>
                    (L -> map(l -> [l, replace(l, "آ" =>
                                   "ا")],
                              L)) |>
                        (L -> vcat(L...)) |>
                            (L -> filter(l -> contains(d["word"], l), L))
           lemmas = split(d["lemma"], "#") |>
                    (L -> map(l -> [l, replace(l, "آ" =>
                                   "ا")],
                              L)) |>
                        (L -> vcat(L...)) |>
                            (L -> filter(l -> contains(d["word"], l), L))
           lemma = lemmas[1]
       end;

       n = length(collect(lemma));
       idx = nothing;
       for i=1:nWord-n+1 # reverse(1:nWord-n+1)
           if join(collect(d["word"])[i:i+n-1], "") == lemma
               idx = i
               break
           end
       end;

       d["suffix"] = join(collect(d["word"])[idx+n:end], "");
       d["res_root"] = d["res"];
       delete!(d, "res");
       d["affix"] = d["suffix"];
       d["data"] = py"""affix_search"""(d["affix"]);
       d["brain"] = "hacktobesurebrainsjump";
       d),
        Dict(:in => ["word"], :out => ["suffix"]))


dicCODE["add it to the beginning of the root's transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = if haskey(d, "res_prefix")
            string(d["res_prefix"], d["res_root"])
        else
            string(d["res"], d["res_root"])
        end; d),
        Dict(:in => ["res_root"], :out => ["res"]))


dicCODE["add it to the end of the root's transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = haskey(d, "res_suffix") ?
            string(d["res_root"], d["res_suffix"]) :
            string(d["res_root"], d["res"]);
         d["res"] = replace(d["res"], "-''"=>"", "-'"=>""); d),
        Dict(:in => ["res_root", "res"], :out => ["res"]))


dicCODE["undo the change to the verb root and use it!"] =
    Functor((d,e=nothing,f=nothing) ->
        (lemmas = split(d["lemma"], "#") |>
            (L -> map(l -> [replace(l, "آ" =>
                        "ا"),
                    replace(l, "ا" =>
                        "آ")], L)) |> (L -> vcat(L...)) |>
                            (L -> filter(l -> contains(d["word"], l), L));
         d["lemma"] = lemmas[1]; d),
            Dict(:in => ["lemma"], :out => ["res"]))


dicCODE["return the concatenation of all the returned transliterations."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = join(d["l_res"], ""); d),
            Dict(:in => ["l_res"], :out => ["res"]))


dicCODE["transliterate it using affix-handler"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         d["res"] = if haskey(d, "prefix")
                        (interfaceName = "affix-handler";
                         node = e[interfaceName];
                         d["affix"]=d["prefix"];
                         if haskey(d, "res")
                             d["res_root"] = d["res"]
                             delete!(d, "res")
                         end;
                         d["data"] = py"""affix_search"""(d["affix"]);
                         d["res_prefix"] = runAgent(node, e, f, d); d)
                    elseif haskey(d, "suffix")
                        (interfaceName = "affix-handler";
                         node = e[interfaceName];
                         d["affix"] = d["suffix"];
                            if haskey(d, "res")
                                d["res_root"] = d["res"]
                                delete!(d, "res")
                            end;
                         d["data"] = py"""affix_search"""(d["affix"]);
                         d["res_suffix"] = runAgent(node, e, f, d); d)
                    end;
            d["brain"] = bN; d),
            Dict(:in => [], :out => ["res"]))

dicCODE["run affix-handler on affix vector"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
        if d["l_affix"] == []
             d["l_res"] = (dd = copy(dataM);
                           dd["word"] = d["affix"];
                           dd["pos"] = d["pos"];
                           dd["brain"] = d["brain"];
                           if haskey(d, "res_root")
                               dd["res_root"] = d["res_root"]
                           end;
                           interfaceName = "terminator";
                           node = e[interfaceName];
                           w = runAgent(node, e, f, dd);
                           [w])

         else
             d["l_res"] = [];
             k = haskey(d, "suffix") ? "suffix" : "prefix";
             segm = nothing;
             for iw in enumerate(d["l_affix"])
                 i = iw[1];
                 w = iw[2];
                 dd = copy(dataN);
                 dd["word"] = w;
                 dd["affix"] = w;
                 dd["brain"] = bN; #d["brain"]; #"affix-handler";
                 dd[k] = w;
                 dd["pos"] = k == "suffix" ?
                    i == 0 ? d["pos"] : "nothing" :
                        i == length(d["l_affix"]) ?
                            d["pos"] : "nothing";
                 dd["segm"] = length(d["l_res"]) == 0 ? nothing : d["l_res"][end];
                 if haskey(d, "res_root")
                     dd["res_root"] = d["res_root"]
                 end;
                 node = e["affix-handler"];
                 push!(d["l_res"],
                       runAgent(node, e, f, dd));
            end;
         end;
          d["brain"]=bN; d),
            Dict(:in => ["l_affix"], :out => ["l_res"]))


dicCODE["find the longest substring of the input that exists in the database."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["d_substring"] = py"""longest_root_and_affixes"""(d["word"]);
         d["word_total"] = d["word"];
         d["data"] = typeof(d["d_substring"]) != String ?
            py"""search_db"""(d["d_substring"]["root"]) : d["d_substring"]; d),
            Dict(:in => ["word"], :out => ["d_substring", "data", "word_total"]))


dicCODE["transliterate each side of it separately in proper order and put its transliteration with the highest frequency between them."] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         d_substrings = d["d_substring"];
         if typeof(d_substrings) == String
             d["res"] = d_substrings
             @goto OUT
         end;
         # root
         root = if d["brain"] == "collision-handler"
             py"""return_highest_search_pos"""(d["data"], d["pos"])
         else
             py"""get_in_db"""(d_substrings["root"], d["pos"])
         end;
         prefix = d_substrings["prefix"];
         suffix = d_substrings["suffix"];
         # prefix and suffix
         prefix = if length(prefix) > 0 && !(prefix == "\u200c")
             dd = copy(dataN);
             dd["word"] = prefix;
             dd["pos"] = d["pos"];
             dd["pre_pos"] = d["pre_pos"];
             interfaceName = "transliterator";
             node = e[interfaceName];
             if haskey(dd, "res")
                 dd["res_root"] = dd["res"]
                 delete!(dd, "res")
             end;
             runAgent(node, e, f, dd);
         else
             ""
         end;
         suffix = if length(suffix) > 0 && !(suffix == "\u200c")
             dd = copy(dataN);
             dd["word"] = suffix;
             dd["pos"] = d["pos"];
             dd["pre_pos"] = d["pre_pos"];
             interfaceName = "transliterator";
             node = e[interfaceName];
             if haskey(dd, "res")
                 dd["res_root"] = dd["res"]
                 delete!(dd, "res")
             end;
             runAgent(node, e, f, dd);
         else
             ""
         end;
         # postprocess
         d["SynCatCode"] = root[2];
         d["root"] = root[1];

         d["res"] = string(prefix, root[1], suffix);
         @label OUT;
         d["brain"] = bN;
         d),
            Dict(:in => ["d_substring"], :out => ["res"]))


dicCODE["move the longest substring of the input that exists in affixes and starts in the beginning of the input to affix vector. if the input is not empty and no substring of the input can be found in affixes, move contents of affix vector back to the input then run terminator on it."] =
    Functor((d,e=nothing,f=nothing) ->
        begin
            if d["affix"] != ""
                d["l_affix"] = py"""recu_affixes_subs"""(d["affix"], d["pos"])
                if !py"""check_all_in_affixes"""(d["l_affix"])
                    d["l_affix"] = []
                end
            end
            d
        end,
            Dict(:in => ["affix"], :out => ["l_affix"]))

dicCODE["update the word's pos according to the database!"] =
    Functor((d,e=nothing,f=nothing) ->
        (synCode = get(d, "SynCatCode", nothing);
         d["pos"] = get(py"""d_map_FLEXI""",synCode, d["pos"]); d),
            Dict(:in => [], :out => ["pos"]))

dicCODE["is there anything after it?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["d_substring"]["suffix"] != "" ? "yes" : "no"; d),
            Dict(:in => ["d_substring", "word_total"], :out => ["state"]))


dicCODE["is there anything before it?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["d_substring"]["prefix"] != "" ?
                "yes" : "no"; d),
            Dict(:in => ["d_substring", "word_total"], :out => ["state"]))

dicCODE["add it to the beginning of its transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["res"]*get(d, "res_root", ""); d),
            Dict(:in => ["res_root", "res"], :out => ["res"]))

dicCODE["add it to the end of its transliteration"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = get(d, "res_root", "")*d["res"]; d),
            Dict(:in => ["res_root", "res"], :out => ["res"]))

dicCODE["is the longest substring, رو recognized as a verb?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["pos"] == "Verb" ? "yes" : "no"; d),
            Dict(:in => [], :out => ["state"]))

dicCODE["transliterate each side of the underscore separately in proper order and add a space between them"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
        d["res"] = map(w -> (dd = copy(dataN);
                              dd["word"] = w;
                              dd["pos"] = d["pos"];
                              interfaceName = "transliterator";
                              node = e[interfaceName];
                              runAgent(node, e, f, dd)),
                    split(d["word"], "_")) |>
                        (D -> join(D, " "));
         d["root"] = d["word"]; # to end computation
         d["brain"] = bN;
         d),
            Dict(:in => ["lemma"], :out => ["res"]))

##############
# MODEL6
##############

dicCODE["move the longest substring of the input that exists in affixes and starts in the beginning of the input to affix vector."] =
    Functor((d,e=nothing,f=nothing) ->
        begin
            if !haskey(d, "input")

                dd = copy(dataM)
                dd["pos"] = d["pos"]
                dd["affix"] = d["affix"]
                dd["input"] = d["affix"]
                dd["res_root"] = d["res_root"]
                if haskey(d, "suffix") || haskey(d, "prefix")
                    k = haskey(d, "suffix") ? "suffix" : "prefix";
                    dd[k] = dd["input"]
                end
                if haskey(d, "hassemispace")
                    dd["hassemispace"] = true
                end
                dd["prefix_vector"] = []
                dd["suffix_vector"] = []

            else

                dd = copy(d)

            end

            idx_l = py"""get_affixes_from_l"""(dd["affix"])
            idx_r = py"""get_affixes_from_r"""(dd["affix"])

            if idx_l > 0

                push!(dd["prefix_vector"], join(collect(dd["input"])[1:idx_l]))
                dd["input"] = join(collect(dd["input"])[idx_l+1:end])

            elseif idx_r > 0 # idx_l > 0 && idx_r == length(dd["input"])

                push!(dd["suffix_vector"], join(collect(dd["input"])[idx_r+1:end]))
                dd["input"] = join(collect(dd["input"])[1:idx_r])

            end

            dd
            
        end,
            Dict(:in => ["affix"],
                 :out => ["input","prefix_vector","suffix_vector"]))

dicCODE["is the input empty"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["input"] == "" ?
                "yes" : "no";
         if d["state"] == "yes"
             d["l_affix"] = [d["prefix_vector"];d["suffix_vector"]]
         end;
         d),
        Dict(:in => ["input","prefix_vector","suffix_vector"],
             :out => ["state"]))


dicCODE["can any substrings of the input be found in affixes?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = py"""is_any_substring_in_affixes"""(d["input"]) ?
            "yes" : "no";
            d),
        Dict(:in => ["input"], :out => ["state"]))


dicCODE["move contents of affix vector back to the input then run transliterator on it."] =
    Functor((d,e=nothing,f=nothing) ->
        begin
            bN = "affix-sub";
            w = join([d["prefix_vector"]; d["input"]; d["suffix_vector"]]);
            k = haskey(d, "suffix") ? "suffix" : "prefix";
            segm = nothing;
            dd = copy(dataN);
            dd["word"] = w;
            dd["affix"] = w;
            dd["brain"] = bN;
            dd[k] = w;
            dd["res_root"] = d["res_root"]
            dd["pos"] = d["pos"];
            node = e["transliterator"];
            d["res"] = runAgent(node, e, f, dd)
            d["brain"] = bN;
            d
        end,
    Dict(:in => ["input","prefix_vector","suffix_vector"],
         :out => ["res"]))

dicCODE["run affix-handler on it"] =
    Functor((d,e=nothing,f=nothing) ->
        (bN = d["brain"];
         if !haskey(d, "l_affix")
             d["l_affix"] = [d["word"]]
         elseif d["l_affix"] == []
             d["l_affix"] == [d["input"]]
         end;
#exit();
         d["l_res"] = [];
         k = haskey(d, "suffix") ? "suffix" : "prefix";
         segm = nothing;
         for iw in enumerate(d["l_affix"])
             i = iw[1];
             w = iw[2];
             dd = copy(dataN);
             dd["word"] = w;
             dd["affix"] = w;
             if get(d, "hassemispace", false)
                 dd["hassemispace"] = d["hassemispace"]
             end
             dd["brain"] = bN;
             dd[k] = w;
             dd["pos"] = k == "suffix" ?
                i == 0 ? d["pos"] : "nothing" :
                    i == length(d["l_affix"]) ?
                        d["pos"] : "nothing";
             dd["segm"] = length(d["l_res"]) == 0 ? nothing : d["l_res"][end];
             node = e["affix-handler"];
             push!(d["l_res"],
                   runAgent(node, e, f, dd));
          end;
          d["brain"]=bN;
          d),
            Dict(:in => [], :out => ["l_res"]))
