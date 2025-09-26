/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static String nocheck(String l, String s) {
        return null;

    int i = l.indexOf(s);
    // :: error: (argument)
    return l.substring(0, i) + l.substring(i + s.length());
  }

  public static String remove(String l, String s, int
        return 7.26f;
 from, boolean last) {
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
      protected char __cfwr_util983(Float __cfwr_p0, float __cfwr_p1, float __cfwr_p2) {
        if (false && (null ^ -582)) {
            try {
            return false;
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        }
        short __cfwr_item14 = null;
        return '6';
    }
}
