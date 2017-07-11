# opt_tools
Tools for logging an optimisation procedure. Specially designed to work with scipy and GPflow.

See the `./examples/` directory to see how to use.

### Use within other repositories
I usually add this as a git subtree:

```
git remote add st-opt_tools git@github.com:markvdw/opt_tools.git
git subtree add --prefix=opt_tools/ st-opt_tools master
```

Updates can be pulled and pushed using:

```
git subtree pull --prefix=opt_tools st-opt_tools master
git subtree push --prefix=opt_tools st-opt_tools master
```

### Changelog
- v2.0 (06/01/2017): Tasks are now separated out from the helper classes.
- v1.0