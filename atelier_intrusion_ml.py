def explore_data(df):
    print("Distribution des classes:")
    print(df['is_intrusion'].value_counts(), "\n")

    print("Proportions:")
    print((df['is_intrusion'].value_counts(normalize=True) * 100).round(2), "%\n")

    print("Statistiques descriptives par classe:")
    print(df.groupby('is_intrusion').describe().transpose(), "\n")

    # Visualisations
    features_to_plot = ['packet_size', 'duration', 'src_bytes', 'dst_bytes', 'num_failed_logins']
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(features_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.histplot(
            data=df,
            x=col,
            hue="is_intrusion",
            kde=True,
            stat="density",
            common_norm=False
        )
        plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.show()

    # Corrélation (hors target)
    plt.figure(figsize=(8, 6))
    corr = df.drop(columns=['is_intrusion']).corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Corrélation des features')
    plt.show()
