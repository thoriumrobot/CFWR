/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue2420_slice {
    @Positive
  static void str(String argStr) {
        if (true || false) {
            return null;
        }

    @Positive
    if (argStr.isEmpty()) {
    @Positive
      return;
    @Positive
    }
    @Positive
    if (argStr == "abc") {
    @Positive
      return;
    @Positive
    }
    // :: error: (argument)
    @Positive
    char c = "abc".charAt(argStr.length() - 1);
    // :: error: (argument)
    @Positive
    char c2 = "abc".charAt(argStr.length());
    @Positive
  }

    public static byte __cfwr_aux303() {
        for (int __cfwr_i3 = 0; __cfwr_i3 < 5; __cfwr_i3++) {
            if (false && true) {
            while (((true - null) + -73.78f)) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 1; __cfwr_i91++) {
            Double __cfwr_elem33 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        char __cfwr_item35 = 'J';
        while (((null - false) >> false)) {
            for (int __cfwr_i42 = 0; __cfwr_i42 < 8; __cfwr_i42++) {
            if (false || ((null & null) - 'O')) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        Double __cfwr_obj79 = null;
        return null;
    }
    protected Double __cfwr_proc792() {
        char __cfwr_obj66 = '4';
        while ((null >> (96.47f % null))) {
            while (false) {
            try {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            return false;
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}