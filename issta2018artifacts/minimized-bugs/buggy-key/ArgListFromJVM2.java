// simplified version of the second of the two bugs fixed in plume-lib rev. 4f10607

public class ArgListFromJVM2 {

    /*@ public normal_behavior
      @ requires true;
      @ ensures true;
      @*/
    public static void arglistFromJvm(String arglist) {
        if (!(arglist.startsWith("(") && arglist.endsWith(")"))) {
            return;
        }
        String result = "(";
        int pos = 1;
        while (pos < arglist.length() - 1) {
            if (pos > 1) {
                result += ", ";
            }
            int nonarray_pos = pos;
            while (arglist.charAt(nonarray_pos) == '[') {
                nonarray_pos++;
                if (nonarray_pos >= arglist.length()) {
                    return;
                }
            }
            char c = arglist.charAt(nonarray_pos);
            if (c == 'L') {
                int semi_pos = arglist.indexOf(";", nonarray_pos);
                result += arglist.substring(pos, semi_pos + 1);
                pos = semi_pos + 1;
            }
        }
    }
}
