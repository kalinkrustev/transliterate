const fs = require('fs');

const words = [...new Set(fs.readFileSync('bg.txt').toString('utf8').split('\n').map(x => x.split('/')[0].toLowerCase()))];

const cyrillic = {
    'а': ['a'],
    'б': ['b'],
    'в': ['v', 'w'],
    'г': ['g'],
    'д': ['d'],
    'е': ['e'],
    'ж': ['zh', 'j'],
    'з': ['z'],
    'и': ['i'],
    'й': ['y', 'j', 'i'],
    'к': ['k'],
    'л': ['l'],
    'м': ['m'],
    'н': ['n'],
    'о': ['o'],
    'п': ['p'],
    'р': ['r'],
    'с': ['s'],
    'т': ['t'],
    'у': ['u'],
    'ф': ['f'],
    'х': ['h', 'x'],
    'ц': ['ts', 'c', 'tz'],
    'ч': ['ch', '4'],
    'ш': ['sh', '6'],
    'щ': ['sht', '6t'],
    'ъ': ['a', 'y', 'u'],
    'ь': ['i', 'j', 'y'],
    'ю': ['yu', 'iu', 'ju', 'u'],
    'я': ['ya', 'ia', 'ja', 'q']
};
const variants = word => word
    .split('')
    .reduce((prev, cur) => [].concat(...cyrillic[cur].map(char => prev.map(word => word + char))), ['']);
const transliterate = require('../ut-transliterate');

const ambiguousWord = word =>
    variants(word)
        .filter(variant => word !== transliterate(variant))
        .reduce((prev, cur) => prev.concat([[cur, word, transliterate(cur)]]), []);

const ambiguous = [].concat(...words.map(ambiguousWord));
let map = ambiguous.reduce((prev, cur) => {
    const word = prev[cur[0]];
    if (word) {
        word.push(cur[1]);
    } else {
        prev[cur[0]] = [cur[1]];
    }
    return prev;
}, {});

console.log(Object.values(map).filter(x => x.length > 1));
console.log(ambiguous.length);

fs.writeFileSync('amb.txt', Object.entries(map).map(([key, value]) => [key, value.join(' ')].join(' ')).join('\n'));

map = ambiguous.reduce((prev, cur) => {
    prev[cur[0]] = cur[1];
    return prev;
}, {});

fs.writeFileSync('../ut-transliterate/bg.json', JSON.stringify(map, true, 2));
