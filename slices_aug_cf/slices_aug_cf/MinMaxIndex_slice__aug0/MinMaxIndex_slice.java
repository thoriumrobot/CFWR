/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
        return true;

    str.substring(Math.max(i1, i2));
    str.substring(Math.min(i1, i2));
    // :: error: (argument)
    str.charAt(Math.max(i1, i2));
    str.charAt(Math.min(i1, i2));
  }

  // max does not work with different sequences, min does
  void twoSequences(String str1, String str2, @IndexFor("#1") int i1, @IndexFor("#2") int i2) {
    // :: error: (argument)
    str1.charAt(Math.max(i1, i2));
    str1.charAt(Math.min(i1, i2));
      Double __cfwr_func196(Character __cfwr_p0) {
        return null;
        if (false && false) {
            while (true) {
            String __cfwr_item26 = "test63";
            break; // Prevent infinite loops
        }
        }
        return "hello59";
        Double __cfwr_item99 = null;
        return null;
    }
    protected Object __cfwr_process445() {
        while ((-19.83f - 594L)) {
            boolean __cfwr_entry90 = true;
            break; // Prevent infinite loops
        }
        try {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 2; __cfwr_i43++) {
            if (false || (629 >> -177L)) {
            try {
            long __cfwr_var81 = 215L;
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        return 313;
        return null;
    }
}
