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
      protected Boolean __cfwr_proc982(byte __cfwr_p0) {
        try {
            String __cfwr_elem5 = "world28";
        } catch (Exception __cfwr_e79) {
            // ignore
        }

        return null;
        try {
            if (true && true) {
            try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 8; __cfwr_i50++) {
            try {
            short __cfwr_node82 = null;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        return null;
    }
    protected static float __cfwr_util434(int __cfwr_p0, int __cfwr_p1) {
        return null;
        if ((468 + null) || false) {
            return -719L;
        }
        try {
            return null;
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        return -45.09f;
    }
}
