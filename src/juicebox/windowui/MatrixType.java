package juicebox.windowui;


public enum MatrixType {
    OBSERVED("Observed"),
    OE("OE"),
    PEARSON("Pearson"),
    EXPECTED("Expected"),
    RATIO("Observed / Control"),
    CONTROL("Control");
    private final String value;

    MatrixType(String value) {
        this.value = value;
    }

    public String toString() {
        return value;
    }

}
