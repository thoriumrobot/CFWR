// version of code that causes bug fixed by plume-lib rev. 339602f

public class Exdent {
    boolean enabled = true;
    String indent_str;
    final String INDENT_STR_ONE_LEVEL = "    ";

    /** Exdents: reduces indentation and pops a start time. */
        /*@ public normal_behavior
@ requires indent_str.length() > INDENT_STR_ONE_LEVEL.length();
@ ensures true;
@*/
    public void exdent() {
	//:: error: (argument.type.incompatible)
	indent_str = indent_str.substring(0, indent_str.length() - INDENT_STR_ONE_LEVEL.length());
    }
}
