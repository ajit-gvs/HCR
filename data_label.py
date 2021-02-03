def data_label():
  samples=[]
  f = open(data_dir /ascii/'words.txt')
   for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')
            assert len(lineSplit) >= 9
            fileNameSplit = lineSplit[0].split('-')
            fileName = data_dir /words/ fileNameSplit[0] / f'{fileNameSplit[0]}-{fileNameSplit[1]}' / lineSplit[0] + '.png'
            gtText = lineSplit[8:]
            chars = chars.union(set(list(gtText)))

            # put sample into list
             samples.append([fileName,gtText])
  return samples


