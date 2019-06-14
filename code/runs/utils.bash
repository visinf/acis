#!/bin/bash

check_rundir()
{
  dir="$1"
  if [ -d $dir ]; then
    read -p "Directory exists: $dir. Delete ('Y' for yes)? " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^Y$ ]]; then
      echo "Exiting"
      exit 1
    else
      rm -rf $dir
      echo "Removed $dir"
    fi
  fi
}
