// simplified test case for bug fixed by plume-lib commit d535562
// This is an odd one, since the Index Checker can't actually handle
// this without a false positive due to not having good list support,
// but we still found and fixed the bug because of the Index Checker.

public class FileCompiler {


    //@ public invariant compiler.length >= 1;

    String [] compiler;

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public FileCompiler(ArrayList compiler) {
	//:: error: assignment.type.incompatible
	this.compiler = compiler.toArray(new String[0]);
    }

    class ArrayList {

        /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
        String[] toArray(String[] a) { return a; }
    }
}
