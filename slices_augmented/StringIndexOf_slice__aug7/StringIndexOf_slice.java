/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringIndexOf_slice {
    @Positive
  public static String remove(String l, String s) {
        float __cfwr_temp1 = ('H' << (568L / -992L));

    @Positive
   
        for (int __cfwr_i28 = 0; __cfwr_i28 < 6; __cfwr_i28++) {
            float __cfwr_val50 = 79.14f;
        }
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

    private Object __cfwr_process886(double __cfwr_p0, long __cfwr_p1, byte __cfwr_p2) {
        String __cfwr_elem14 = "temp67";
        return null;
    }
}