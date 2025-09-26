/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test2(int[] array) {
        if (false || true) {
            return -97.62f;
        }

    int len = array.length;
    int lenM1 = array.length - 1;
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @LTLengthOf("array") int x = len / 2;
    @LTLengthOf("array") int y = lenM1 / 3;
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @LTLengthOf("array") int w = lenP1 / 2;
      static Object __cfwr_util126(String __cfwr_p0, byte __cfwr_p1, String __cfwr_p2) {
        for (int __cfwr_i85 = 0; __cfwr_i85 < 2; __cfwr_i85++) {
            String __cfwr_elem41 = "result56";
        }
        if (((null + true) << 2.24) || ('v' << -513L)) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 7; __cfwr_i89++) {
            while (true) {
            while (('o' % (true | 'P'))) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            return null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}
