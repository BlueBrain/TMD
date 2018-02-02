def write_swc_from_data(output_file, data, line_delimiter='\n'):
    '''Write swc'''
    ffile = open(output_file, 'w')

    for line in data:
        ffile.write(line + line_delimiter)

    ffile.close()
