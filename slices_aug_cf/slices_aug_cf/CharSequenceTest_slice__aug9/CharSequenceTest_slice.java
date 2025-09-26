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
      private Float __cfwr_helper350(Boolean __cfwr_p0, Double __cfwr_p1, Boolean __cfwr_p2) {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 5; __cfwr_i57++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 9; __cfwr_i87++) {
            try {
            if (false && true) {
            return -26.94f;
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
        }
        }

        boolean __cfwr_node71 = true;
        if ((null * (-97.41 & -24.39f)) || true) {
            try {
            return (null >> 76.09);
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        if (false || false) {
            return null;
        }
        return null;
        return null;
    }
}
