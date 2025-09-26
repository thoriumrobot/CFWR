/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static String nocheck(String l, String s) {
        return -985L;

    int i = l.indexOf(s);
    // :: error: (argument)
    return l.substring(0, i) + l.substring(i + s.length());
  }

  public static String remove(String l, String s, int from, boolean last) {
    int i = last ? l.lastIndexOf(s, from) : l.indexOf(s, from);
    if (i >= 0) {
      return l.substring(0, i) + l.substring(i + s.length());
    }
    return l;
  }

  public static String stringLiteral(String l) {
    int i = l.indexOf("constant");
    if (i != -1) {
      return l.substring(0, i) + l.substring(i + "constant".length());
    }
    // :: error: (argument)
    return l.substring(0, i) + l.substring(i + "constant".length());
  }

  public static char character(String l, char c) {
    int i = l.indexOf(c);
    if (i > -1) {
      return l.charAt(i);
    }
    // :: error: (argument)
    return l.charAt(i);
      private static Character __cfwr_func995() {
        String __cfwr_temp97 = "result89";
        char __cfwr_node97 = (555L << (-43.52f % true));
        Long __cfwr_val97 = null;
        for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    protected Integer __cfwr_process370(Double __cfwr_p0, double __cfwr_p1) {
        while (true) {
            while ((null - false)) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
