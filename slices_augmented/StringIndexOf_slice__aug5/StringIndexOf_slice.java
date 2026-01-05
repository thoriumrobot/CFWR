/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringIndexOf_slice {
    @Positive
  public static String remove(String l, String s) {
        return null;

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

    public static Character __cfwr_helper47(byte __cfwr_p0, Character __cfwr_p1) {
        return 853;
        return null;
        if (false || true) {
            while (false) {
            while (true) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 3; __cfwr_i76++) {
            try {
            return null;
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        try {
            byte __cfwr_val96 = ((-38.75 * false) << ('E' + 'O'));
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        return null;
    }
    protected static float __cfwr_temp212(String __cfwr_p0) {
        try {
            try {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 5; __cfwr_i56++) {
            if (true && (null >> 22.29f)) {
            Object __cfwr_item25 = null;
        }
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        return (-0.71f / (-26.56 ^ 724L));
    }
    private String __cfwr_compute577(Integer __cfwr_p0, byte __cfwr_p1, byte __cfwr_p2) {
        try {
            try {
            return 'e';
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        return "result14";
    }
}