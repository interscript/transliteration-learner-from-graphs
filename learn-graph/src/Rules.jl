
# https://lucid.app/lucidchart/008ff36b-136f-4666-b06e-ac3b2a8e8d85/edit?page=0_0&invitationId=inv_0221b3ba-bd16-4499-8f60-b95aa68af45c

# Read Mappings
dISOMapping = map(x -> split(x), #|> println,
    readlines("resources/Strict+ShortVowelsOnlyIf.txt")) |>
    (D -> filter(x ->  length(x) > 1, D)) |>
    (D -> map(d -> collect(d[1]) => d[2], D)) |>
    (D -> filter(d -> length(d[1]) == 1, D)) |>
    (D -> map(d -> d[1][1] => d[2], D)) |>
                Dict


# Computations state
dataSTATE = Dict{String, Any}(
            "state" => nothing,
            "brain" => nothing);


# Dictionary with commands
dicCODE = Dict{String, Functor}()


# Node terminating computation

dicCODE["keep running mapper till the input ends!"] =
    Functor((d,e=nothing,f=nothing) ->
        begin

            dd = copy(dataSTATE)
            dd["v_chars"] = collect(d["txt"])
            dd["n_chars"] = length(dd["v_chars"])
            dd["ix"] = 1
            dd["transliteration"] = ""
            transliteration = []

            while dd["ix"] < dd["n_chars"]+1

                interfaceName = "mapper"
                node = e[interfaceName]
                id = dd["ix"]
                res = runAgent(node, e, f, dd)
                @info "@@@@@@@", dd["v_chars"][id:dd["ix"]-1], " ===> ", res
                delete!(dd, "res")
                dd["transliteration"] = dd["transliteration"] * res

            end

            d["res"] = dd["transliteration"] #join(transliteration) #dd["transliteration"]
            d

        end,
        Dict(:in => ["txt"], :out => ["transliteration"]))

dicCODE["return the character!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = d["v_chars"][d["ix"]];
         d["ix"] = d["ix"] + 1; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["return its transliteration!"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = dISOMapping[d["v_chars"][d["ix"]]];
         d["ix"] = d["ix"] + 1;
         d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["is the character found in the mapping file?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = haskey(dISOMapping, d["v_chars"][d["ix"]]) ? "yes" : "no";
         d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["v_chars", "n_chars", "ix"]))

# is it ?
dicCODE["is it ا?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"][d["ix"]] == 'ا' ? "yes" : "no"; d),
            Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is it ە?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"][d["ix"]] == 'ە' ? "yes" : "no"; d),
            Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is it ه?"] =
Functor((d,e=nothing,f=nothing) ->
    (d["state"] = d["v_chars"][d["ix"]] == 'ه' ? "yes" : "no"; d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is it ّ?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] = d["v_chars"][d["ix"]] == 'ّ' ? "yes" : "no"; d),
            Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))


# next characters
dicCODE["is the next character ٔ?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if d["n_chars"] > d["ix"]
                d["v_chars"][d["ix"]+1] == 'ٔ' ? "yes" : "no";
            else
                "no"
            end; d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is the next character ل?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if d["n_chars"] > d["ix"]
                d["v_chars"][d["ix"]+1] == 'ل' ? "yes" : "no";
            else
                "no"
            end; d),
            Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is the 2nd next character ل?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if d["n_chars"] > d["ix"]+1
                d["v_chars"][d["ix"]+2] == 'ل' ? "yes" : "no";
            else
                "no"
            end; d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["is the 3rd next character ه?"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["state"] =
            if d["n_chars"] > d["ix"]+2
                d["v_chars"][d["ix"]+3] == 'ه' ? "yes" : "no";
            else
                "no"
            end; d),
        Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["state"]))

dicCODE["return allāh for these four characters and pass the 4th next character to mapper"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "allāh";
         d["ix"] = d["ix"]+4; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["return â for this character, and pass the next character to mapper"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "â";
         d["ix"] = d["ix"]+1; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["return âl for these two characters, and pass the 2nd next character to mapper"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "âl";
         d["ix"] = d["ix"]+2; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["return âll for these three characters, and pass the 3rd next character to mapper"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "âll";
         d["ix"] = d["ix"]+3; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["repeat last character's transliteration and underline both."] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = string(d["transliteration"][end], '͟',
                           d["transliteration"][end]);
         d["transliteration"] = join(collect(d["transliteration"])[1:end-1]);
         d["ix"] = d["ix"]+1; d),
        Dict(:in => ["transliteration", "v_chars", "ix"],
             :out => ["res", "ix", "transliteration"]))

dicCODE["return h for this character, and pass the next character to mapper"] =
Functor((d,e=nothing,f=nothing) ->
    (d["res"] = "h";
     d["ix"] = d["ix"]+1; d),
     Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))

dicCODE["return h’ for these two characters, and pass the 2nd next character to mapper"] =
    Functor((d,e=nothing,f=nothing) ->
        (d["res"] = "h’";
         d["ix"] = d["ix"]+2; d),
         Dict(:in => ["n_chars", "v_chars", "ix"], :out => ["res", "ix"]))


dicCODE["change all instances of ي and ك and ۀ in the text to ی and ک and هٔ "] =
    Functor((d,e=nothing,f=nothing) ->
        begin
            d["txt"] = replace(d["txt"], 'ك'=>
                                     'ک',
                                        'ي'=>
                                        'ی',
                                           "ۀ" =>
                                            "هٔ")
            d
        end,
        Dict(:in => ["txt"], :out => ["txt"]))

#===
dicCODE["change all instances of ي and ك and ۀ and ۹ in the text to ی and ک and هٔ and ٩"] =

    Functor((d,e=nothing,f=nothing) ->
        begin
            d["txt"] = replace(d["txt"], 'ك'=>
                                         'ک',
                                            'ي'=>
                                            'ی',
                                               "ۀ" =>
                                                "هٔ",
                                            '۹' =>
                                            '٩')
            d
        end,
        Dict(:in => ["txt"], :out => ["txt"]))
===#
