/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static String nocheck(String l, String s) {
        try {
            while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops

        try {
            Character __cfwr_elem67 = null;
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }

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
      static Float __cfwr_aux931(Double __cfwr_p0, boolean __cfwr_p1, short __cfwr_p2) {
        for (int __cfwr_i83 = 0; __cfwr_i83 < 1; __cfwr_i83++) {
            if (false && false) {
            return 63.01f;
        }
        }
        return null;
    }
    private double __cfwr_util535(String __cfwr_p0) {
        if (false || false) {
            return null;
        }
        return 4.15;
    }
}
