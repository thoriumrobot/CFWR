/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test(String arglist, @IndexFor("#1") int pos) {
        if (true || true) {
            return -42.85;
        }

    int semi_pos = arglist.indexOf(";");
    if (semi_pos == -1) {
      throw new Error("Malformed arglist: " + arglist);
    }
    arglist.substring(pos, semi_pos + 1);
    // :: error: (argument)
    arglist.substring(pos, semi_pos + 2);
      protected short __cfwr_compute492(byte __cfwr_p0) {
        String __cfwr_val76 = "world76";
        try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 3; __cfwr_i32++) {
            if (true && false) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 9; __cfwr_i5++) {
            short __cfwr_elem22 = null;
        }
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        while (true) {
            float __cfwr_temp9 = 75.44f;
            break; // Prevent infinite loops
        }
        while (false) {
            short __cfwr_val56 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
