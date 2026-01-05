/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringIndexOf_slice {
    @Positive
  public static String remove(String l, String s) {
        while ((true / null)) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 7; __cfwr_i17++) {
            short __cfwr_item46 = (4.39f << (52.28 | null));
        }
            break; // Prevent infinite loops
        }

    @Positive
    int i = l.indexOf(s);
    @Positive
    if (i != -1) {
    @Positive
      return l.substring(0, i) + l.substring(i + s.length());
    @Positive
    }
    @Positive
    return l;
    @Positive
  }

    @Positive
  public static String nocheck(String l, String s) {
    @Positive
    int i = l.indexOf(s);
    // :: error: (argument)
    @Positive
    return l.substring(0, i) + l.substring(i + s.length());
    @Positive
  }

    @Positive
  public static String remove(String l, String s, int from, boolean last) {
    @Positive
    int i = last ? l.lastIndexOf(s, from) : l.indexOf(s, from);
    @Positive
    if (i >= 0) {
    @Positive
      return l.substring(0, i) + l.substring(i + s.length());
    @Positive
    }
    @Positive
    return l;
    @Positive
  }

    public static long __cfwr_aux790(int __cfwr_p0, Object __cfwr_p1) {
        Object __cfwr_entry20 = null;
        return -79L;
    }
    private Boolean __cfwr_temp32() {
        for (int __cfwr_i79 = 0; __cfwr_i79 < 7; __cfwr_i79++) {
            for (int __cfwr_i99 = 0; __cfwr_i99 < 2; __cfwr_i99++) {
            try {
            try {
            if (true || false) {
            try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        }
        Boolean __cfwr_node26 = null;
        return null;
    }
}