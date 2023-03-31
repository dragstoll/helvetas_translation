import pandas as pd
import numpy as np

DATA_FOLDER = '../Daten_pcm_export/'

def load():
    file_list = ['klienten_pcm.csv', 'faelle_pcm.csv' , 'massnahmen_pcm.csv','produkte_pcm.csv', #  'faelle2_pcm_20210919.csv'
             'platzierungen_pcm.csv','aufsuchende_familienarbiet_pcm.csv','tagesstruktur_pcm.csv',
                'mutter_export_20210819.csv','vater_export_20210819.csv']
    df = {file[:3]:pd.read_csv(DATA_FOLDER+file,delimiter=';',skiprows=[1]) for file in file_list}

    preprocess_kli(df['kli'])
    preprocess_fae(df['fae'])
    preprocess_mas(df['mas'])
    preprocess_eltern(df['mut'],'MUTTER')
    preprocess_eltern(df['vat'],'VATER')

    
    df['kli']['Nmas'] = key_count(df['kli'],df['mas'],'CDW_KLIENT_ID')
    df['kli']['Npla'] = key_count(df['kli'],df['pla'],'CDW_KLIENT_ID')
    df['kli']['Nauf'] = key_count(df['kli'],df['auf'],'CDW_KLIENT_ID')
    df['kli']['Ntag'] = key_count(df['kli'],df['tag'],'CDW_KLIENT_ID')
    df['kli']['Nfae'] = key_count(df['kli'],df['fae'],'CDW_KLIENT_ID')
    df['kli']['Npro'] = key_count(df['kli'],df['pro'],'CDW_KLIENT_ID')
    df['kli']['Nbetr'] = df['kli']['Npla'] + df['kli']['Nauf'] + df['kli']['Ntag']
    df['kli']['Nmut'] = key_count(df['kli'],df['mut'],'CDW_SOURCE_KEY')
    df['kli']['Nvat'] = key_count(df['kli'],df['vat'],'CDW_SOURCE_KEY')
    df['kli']['Neltern'] = df['kli']['Nvat'] + df['kli']['Nmut']
    
    return df

# ------
def preprocess_fae(df):
    df['FALL_AUFNAHME_DATUM'] = df['FALL_AUFNAHME_DATUM'].apply(parse_date)
    df['FALL_AUFNAHME_Jahr'] = df['FALL_AUFNAHME_DATUM'].apply(get_year)

# ------
def preprocess_kli(df):
    df['KLIENT_GEBURTSDATUM'] = df['KLIENT_GEBURTSDATUM'].apply(parse_date)
    df['KLIENT_Geburtsjahr'] = df['KLIENT_GEBURTSDATUM'].apply(get_year)
    df['PLZ_KLIENT_WOHNORT'] = df['PLZ_KLIENT_WOHNORT'].apply(parse_plz)

# ------
def preprocess_eltern(df, tag):
    df['CDW_SOURCE_KEY']= df['ID_K_ELTERN_FK'].apply(lambda s: s.strip(' ')) 
    df['GEBURTSDATUM_YYMD_'+tag] = df['GEBURTSDATUM_YYMD_'+tag].apply(parse_date).astype(np.datetime64)
    df['Geburtsjahr_'+tag] = df['GEBURTSDATUM_YYMD_'+tag].apply(get_year)
    df['PLZ_GESETZ_WOHNSITZ_'+tag] = df['PLZ_GESETZ_WOHNSITZ_'+tag].apply(parse_plz)

# ------
def preprocess_mas(df,prefix = 'zgb'):
    ZGBdata = [['308 Abs. 1 + 2 (+3)',
          '308',
          '308a321',
          'Beistandschaft Rat und Tat, besondere Befugnisse, elterliche Sorge entsprechend beschränkt'],
         ['308 Abs. 1 + 2',
          '308',
          '308a21',
          'Erziehungsbeistandschaft und besondere Befugnisse'],
         ['308 Abs. 2', '308', '308a2', 'Beistandschaft, besondere Befugnisse'],
         ['308 Abs. 1', '308', '308a1', 'Erziehungsbeistandschaft'],
         ['306 Abs. 2', '306', '306', 'Interessenkollision/Verhinderung der Eltern'],
         ['* 309 / 308', '308', '308u09', 'Beistandschaft Vaterschaft/Unterhalt'],
         ['310 / 308',
          '310',
          '310u08',
          'Aufhebung des Aufenthaltsbestimmungsrechts/Beistandschaft'],
         ['307', '307', '307', 'Kindesschutz (Weisung, Erziehungsaufsicht)'],
         ['327a (311/312/298 Abs. 2)',
          '327',
          '327',
          'Minderjährige unter Vormundschaft'],
         ['308 Abs. 2',
          '308',
          '308a2v',
          'Beistandschaft Regelung Vaterschaft/Unterhalt'],
         ['310 / 308 / 314b',
          '310',
          '310u08u14',
          'Aufhebung des Aufenthaltsbestimmungsrechts/Beistandschaft/FU'],
         ['299 / 300 ZPO',
          'ZPO',
          'ZPO',
          'Verfahrensvertretung in familienrechtlichen Verfahren'],
         ['Art. 18 BG-HAÜ', 'BG', 'BG18', 'Adoptionsvormundschaft'],
         ['308 Abs. 2', '308', '308a2u', 'Beistandschaft Regelung Unterhalt'],
         ['324 / 325',
          '324',
          '324',
          'Schutz des Kindesvermögens (Aufsicht/Beistandschaft)'],
         ['314abis Abs.2 Ziff.2', '314', '314', 'Verfahrensbeistandschaft vor KESB'],
         ['Art. 17 BG-HAÜ', 'BG', 'BG17', 'Adoptionsbeistandschaft'],
         ['308 Abs. 2 + 3',
          '308',
          '308a32',
          'Beistandschaft, besondere Befugnisse, elterliche Sorge entsprechend beschränkt']]

    ZGBdf = (pd.DataFrame(ZGBdata, columns=['ZGBARTIKEL','ZGBkey1','ZGBkey2','ZGBTEXTformatiert'])
             .set_index('ZGBTEXTformatiert')
             .sort_values(by='ZGBkey2') )
                     
    df['ZGBTEXTformatiert'] = df.ZGBTEXT.apply(lambda s: s.strip(' ').replace(' / ','/'))
    df['ZGBkey1'] = ZGBdf['ZGBkey1'].reindex(df['ZGBTEXTformatiert']).values
    df['ZGBkey2'] = ZGBdf['ZGBkey2'].reindex(df['ZGBTEXTformatiert']).values
    

# ============== helpers ================================================================   
import datetime
def parse_date(x, cutyear = 2022):
    if np.isreal(x):
        if np.isnan(x):
            x = ''
        else:
            x = str(int(x))
        
    if (len(x) == 8):
        date = np.datetime64('-'.join([x[:4],x[4:6],x[-2:]]))
        if date.astype(object).year <= cutyear:
            return date
        else:
            return np.datetime64('nat')
    else:
        return np.datetime64('nat')
    
def get_year(d):
    try:
        if type(d) == np.datetime64:
            return d.astype(object).year  
        else:
            return int(d.year)
    except:
        return -1  
    
def parse_plz(x):
    # PLZ are strings - especially for foreign countries: DL-12345
    if type(x) == str:
        return x
    else:
        if np.isnan(x):
            return ''
        else:
            return str(x)

    
    
# ============== aggregation functions ==================================================

def one_count(df,column, count_col_name='Counts', 
              percentage = True, percentage_col_name = 'Percentage',minpct=0):
    '''
    one_count(df,column, count_col_name='Counts', 
              percentage = True, percentage_col_name = 'Percentage')
    '''
    
    out = (df.assign(**{count_col_name:1})
            .groupby([column])[[count_col_name]].count()
            .fillna(0).astype(int)
            .sort_values(by=count_col_name,ascending = False) )
    if percentage:
        out[percentage_col_name] = (100*out[count_col_name] / out[count_col_name].sum() ).round(1)
        out = out[out[percentage_col_name] >= minpct]
    
    return out

def two_count(df,index_col, column_col, dense=False):
    out = (df.assign(counts=1)
            .groupby([index_col,column_col])[['counts']].count()
            .reset_index()
            .pivot(index=index_col, columns=column_col,values='counts')
            .fillna(0).astype(int) )
    if dense:
        s = out.stack()
        out = s[s>0].sort_values(ascending=False).reset_index().rename(columns={0:'Counts'})
    return out

def key_count(target_df,count_df,key):
    return (count_df.assign(counts=1)
            .groupby(key)[['counts']].count()
            .reindex(target_df[key]).fillna(0).astype(int)['counts'].values) 

def group_col_to_list(df,groupby,values,sortby=None,joinfun=None):
    '''
    group_col_to_list(df,groupby,values,sortby=None,joinfun=None)
    
    groups a column into lists and returns a series objct
    
    Arguments
    ---------
    
    df : DataFrame
    
    groupby : str
        column name of df for groupby
        
    values : str
        column name of df for values
        
    sortby : str, default = None
        column name of df to sort the values
        
    joinfun : function handle, default = None
        function which takes the list as an input and outputs something else (e.g. concatenated string)
    '''

    def applyfun(df):
        if sortby is not None:
            df = df.sort_values(sortby)
        vallist = df[values].values.tolist()
        if joinfun is None:
            return vallist
        else:
            return joinfun(vallist)
        
    return df.groupby(groupby).apply(applyfun)

