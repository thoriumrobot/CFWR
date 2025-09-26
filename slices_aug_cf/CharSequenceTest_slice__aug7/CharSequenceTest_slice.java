/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testAppend(Appendable app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
  }

  void testAppend(StringWriter app, CharSequence cs, @IndexFor("#2") int i) throws IOException {
    app.append(cs, i, i);
    // :: error: (argument)
    app.append(cs, 1, 2);
      private static String __cfwr_proc937(byte __cfwr_p0) {
        try {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 5; __cfwr_i12++) {
            return null;
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }

        try {
            while ((-492L * false)) {
            short __cfwr_result42 = (false * -843);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        if (true && false) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 7; __cfwr_i17++) {
            return null;
        }
        }
        return "result8";
    }
}
