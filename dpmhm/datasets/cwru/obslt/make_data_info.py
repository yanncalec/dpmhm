# Convert the file `metadata.csv` to a more convenient format.
# Run once.

df = pd.read_csv('cwru/metadata.csv')

mc = {1797:0, 1772:1, 1750:2, 1730:3}

foo = []
for _,rr in df.iterrows():
    if rr['Fault_Location']=='NormalBaseline':
        _sr = 12000
        _loc = None
        _diam = None
        _comp = None
    else:
        _sr = int(rr['Fault_Location'][:2])*1000
        _loc = rr['Fault_Location'][2:-5]
        _diam, _comp = rr['Fault_Type'].split('-')

    foo.append([_sr, mc[rr['RPM']], rr['RPM'], _loc, _comp, _diam, rr['URL']])

dg = pd.DataFrame.from_records(foo, columns=['SamplingRate', 'Load', 'RPM', 'Location',\
                                             'Component', 'Diameter', 'URL'])

dg.to_csv('cwru/data_info.csv', index=False)