

include("get_assets.jl")


py"""

d_corrects = {'ي' : 'ی',
              'ك' : 'ک',
               'ۀ' : 'هٔ'}

d_corrects = dict([(d[0],d[1]) for d in d_corrects.items()])

def normalise(text):
    text = ''.join([d_corrects.get(s, s) for s in list(text)])
    return normalizer.normalize(text)

def search_db(wrd, pos_pos=None):

    # all matches
    l_search = df_Entries[df_Entries['WrittenForm']==wrd].to_dict('records')

    if len(l_search) == 0: # if not found, returns wrd
        return wrd
    else: # if one unique found, return the form
        return l_search

def get_in_db(wrd, pos_pos=None):

    # all matches
    l_search = df_Entries[df_Entries['WrittenForm']==wrd].to_dict('records')

    if len(l_search) == 0: # if not found, returns wrd
        return wrd
    else: # if one unique found, return the form
        return return_highest_search_pos(l_search, pos_pos)

def affix_search(affix, pos_pos=None):

    if affix == '':
        return ''

    l_search = df_Affixes[df_Affixes['Affix']==affix].to_dict('records')

    if len(l_search) == 0: # if not found, returns affix
        return affix
    else:
        return l_search

def get_in_affixes(affix, pos_pos=None):

    # all matches
    l_search = df_Affixes[df_Affixes['Affix']==affix].to_dict('records')
    if len(l_search) == 0: # if not found, returns wrd
        return affix
    else: # if one unique found, return the form
        return return_highest_search_pos(l_search, pos_pos)


def has_entries_search_pos(l_search, pos):
    for d in l_search:
        if d_map_FLEXI.get(d['SynCatCode'], False)==pos:
            return "yes"
    return "no"

def has_only_one_search_pos(l_search, pos=None):

    if type(l_search)==str:
        return "no"

    if not pos is None:
        l_search_tmp = [d for d in l_search
                        if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    else:
        l_search_tmp = l_search

    return "yes" if len(set([d['PhonologicalForm'] for d in l_search_tmp])) == 1 else "no"

def votation_entries(l_search, entries=True):

    d_results, c_results = {}, {}
    for item in l_search:
        form = item['PhonologicalForm']
        if d_results.get(form, False):
            d_results[form] += item.get('Freq', 1)
        else:
            d_results[form] = item.get('Freq', 1)
            c_results[form] = item.get('SynCatCode', "")

    M = max(d_results.values())
    idx = list(d_results.values()).index(M)

    return list(d_results.items())[idx][0], list(c_results.items())[idx][1]


def return_highest_search_pos(l_search, pos):

    if type(l_search) == str:
        return l_search

    data = [d for d in l_search
             if d_map_FLEXI.get(d['SynCatCode'], False)==pos]
    if len(data) == 0:
        return votation_entries(l_search)
    else:
        return votation_entries(data)


def filter_search(l_search, pos_pos=None, pos_neg=None):

    if not pos_pos is None: # filter pos_pos only
        l_search_tmp = [d for d in l_search
                        if d_map_FLEXI.get(d['SynCatCode'], False)==pos_pos]
        if len(l_search_tmp) > 0:
            l_search = l_search_tmp

    if not pos_neg is None: # filter pos_neg out
        l_search = [d for d in l_search
                    if d_map_FLEXI.get(d['SynCatCode'], False)!=pos_neg]

    return l_search


def longest_root_and_affixes(wrd):

    n = len(wrd)
    w_max, l_max = '', 0
    idces_ij = None
    for i in range(n-1):
        #for j in range(i, n):
        for j in range(i, n+1):
            w = wrd[i:j]
            if len(w) > l_max:
                if df_Entries[df_Entries['WrittenForm']==w].shape[0] > 0:
                    idces_ij = (i,j)
                    w_max = w
                    l_max = len(w_max)

    if idces_ij is None:
        return wrd
    else:
        i,j = idces_ij
        return {'root': wrd[i:j], 'prefix': wrd[:i], 'suffix': wrd[j:]}



def recu_entries(wrd, pos_pos=None, pos_neg=None):
    #
    #    Recursive search in entries_DB:
    #    decompose wrd into largest substrings found in DB.
    #
    for i in range(len(wrd), 0, -1):
        if df_Entries[df_Entries['WrittenForm']==wrd[:i]].shape[0] > 0:
            l_search = df_Entries[df_Entries['WrittenForm']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            vota = votation_entries(l_search)

            if wrd[i:] != '':
                return [vota[0]] + recu_affixes(wrd[i:])
            else:
                return [vota[0]]

    return wrd


def recu_affixes(wrd, pos_pos=None, pos_neg=None):
    #
    #    Recursive search in affixes_DB:
    #    decompose wrd into largest substrings found in DB.
    #
    for i in range(len(wrd), 0, -1):
        if df_Affixes[df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            l_search = df_Affixes[df_Affixes['Affix']==wrd[:i]].to_dict('records')
            l_search = filter_search(l_search, pos_pos, pos_neg)
            vota = votation_entries(l_search, entries=False)
            if wrd[i:] != '':
                recu = recu_affixes(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
                return [vota[0]] + recu
            else:
                return [vota[0]]
            break

    return [wrd]

def recu_affixes_subs(wrd, pos_pos=None, pos_neg=None):
    #
    #    Recursive search in affixes_DB:
    #    decompose wrd into largest substrings found in DB.
    #
    for i in range(len(wrd), 0, -1):
        w = wrd[:i]
        if df_Affixes[df_Affixes['Affix']==w].shape[0] > 0:
            if wrd[i:] != '':
                recu = recu_affixes_subs(wrd[i:], pos_pos=pos_pos, pos_neg=pos_neg)
                return [w] + recu
            else:
                return [w]
            break

    return [wrd]

def check_all_in_affixes(l_ws):
    for w in l_ws:
        if df_Affixes[df_Affixes['Affix']==w].shape[0] == 0:
            return False
    return True

def get_affixes_from_l(wrd, pos_pos=None, pos_neg=None):
    # search from left
    idx = 0
    for i in range(len(wrd), 0, -1):
        if df_Affixes[df_Affixes['Affix']==wrd[:i]].shape[0] > 0:
            idx = i
            break
    if idx == 0:
        return idx
    else:
        return idx + 1 # + 1 for julia

def get_affixes_from_r(wrd, pos_pos=None, pos_neg=None):
    # search from right
    idx = len(wrd)-1
    for i in range(0,len(wrd)):
        if df_Affixes[df_Affixes['Affix']==wrd[i:]].shape[0] > 0:
            idx = i
            break
    return idx + 1 # + 1 for julia

def is_any_substring_in_affixes(wrd):

    n = len(wrd)
    w_max, l_max = '', 0
    idces_ij = None
    for i in range(n-1):
        for j in range(i, n+1):
            w = wrd[i:j]
            if len(w) > l_max:
                if df_Affixes[df_Affixes['Affix']==w].shape[0] > 0:
                    return True
    return False

"""
