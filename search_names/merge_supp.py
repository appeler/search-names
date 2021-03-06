#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv

DEFAULT_OUTPUT = "augmented_clean_names.csv"
DEFAULT_NAME_LOOKUP = "FirstName"
DEFAULT_PREFIX_LOOKUP = "seat"
PREFIX_FILE = "prefixes.csv"
NICK_NAMES_FILE = "nick_names.txt"

def parse_command_line():
    """Parse command line options
    """
    parser = argparse.ArgumentParser(description="Merge supplement data")

    parser.add_argument('input', help='Input file name')

    parser.add_argument("-o", "--out", type=str, dest="outfile",
                        default=DEFAULT_OUTPUT,
                        help="Output file in CSV (default: {0!s})"
                        .format(DEFAULT_OUTPUT))
    parser.add_argument("-p", "--prefix", type=str, dest="prefix",
                        default=DEFAULT_PREFIX_LOOKUP,
                        help="Name of column use for prefix look up\
                        (default: {0!s})".format(DEFAULT_PREFIX_LOOKUP))
    parser.add_argument("-n", "--name", type=str, dest="name",
                        default=DEFAULT_NAME_LOOKUP,
                        help="Name of column use for nick name look up\
                        (default: {0!s})".format(DEFAULT_NAME_LOOKUP))
    return parser.parse_args()


def load_prefixes(filename, col):
    prefixes = {}

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            prefixes[r[col]] = r['prefixes']
    return prefixes


def load_nick_names(filename):
    nick_names = {}
    with open(filename) as f:
        for l in f:
            l = l.strip().lower()
            if len(l): # null string
                continue
            a = l.split('-')
            if len(a) >= 2:
                names = a[0].split(',')
                names = [s.strip() for s in names]
                nicks = a[1].split(',')
                nicks = [s.strip() for s in nicks]
                nicks = ';'.join(nicks)
                for n in names:
                    if n in nick_names:
                        print("WARNING: duplicate nick name '{0!s}' for "
                              "'{1!s}'".format(nicks, n))
                    else:
                        nick_names[n] = nicks
            else:
                print("WARNING: Invalid nick name line '{0!s}'".format(l))
    return nick_names

def merge_supp(infile = None, prefixarg = DEFAULT_PREFIX_LOOKUP, name = DEFAULT_NAME_LOOKUP, outfile = DEFAULT_OUTPUT):
    """Merge supplement data to names file
    """
    try:
        f = None
        o = None
        f = open(infile, 'r')
        reader = csv.DictReader(f)
        o = open(outfile, 'w')
        writer = csv.DictWriter(o, fieldnames=reader.fieldnames +
                                ['prefixes', 'nick_names'])
        writer.writeheader()

        for i, r in enumerate(reader):
            print("#{0}: {1!s}".format(i, r[name].lower()))
            # Prefix
            k = r[prefixarg]
            if k in prefixes:
                prefix = prefixes[k]
            else:
                prefix = ''
            r['prefixes'] = prefix
            # Nick name
            k = r[name].lower()
            if k in nick_names:
                nick = nick_names[k]
            else:
                nick = ''
            r['nick_names'] = nick
            writer.writerow(r)
    except Exception as e:
        print(e)
    finally:
        if o:
            o.close()
        if f:
            f.close()

    print("Done.")

if __name__ == "__main__":

    args = parse_command_line()

    prefixes = load_prefixes(PREFIX_FILE, args.prefix)

    nick_names = load_nick_names(NICK_NAMES_FILE)

    print("Merging to '{0!s}', please wait...".format(args.outfile))

    merge_supp(args.input, args.name, args.prefix, args.outfile)
